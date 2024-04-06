import math
from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


import random
from collections import defaultdict

import networkx as nx
from ai2thor.controller import Controller
from ai2thor.util import metrics
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.system import get_logger
from shapely.geometry import Point, Polygon
from procthor_objectnav.utils.types import Vector3


def position_dist(
    p0: Vector3,
    p1: Vector3,
    ignore_y: bool = False,
    dist_fn: Literal["l1", "l2"] = "l2",
) -> float:
    """Distance between two points of the form {"x": x, "y": y, "z": z}."""
    if dist_fn == "l1":
        return (
            abs(p0["x"] - p1["x"])
            + (0 if ignore_y else abs(p0["y"] - p1["y"]))
            + abs(p0["z"] - p1["z"])
        )
    elif dist_fn == "l2":
        return math.sqrt(
            (p0["x"] - p1["x"]) ** 2
            + (0 if ignore_y else (p0["y"] - p1["y"]) ** 2)
            + (p0["z"] - p1["z"]) ** 2
        )
    else:
        raise NotImplementedError(
            'dist_fn must be in {"l1", "l2"}.' f" You gave {dist_fn}"
        )


def get_room_connections(house) -> Tuple[nx.Graph, Dict[int, int]]:
    """Return a graph of how rooms are connected in a house.

    The room_id_to_open_key substitutes the room_id's of neighboring rooms
    that don't share a wall to be together.
    """
    open_wall_connections = defaultdict(set)
    for wall in house["walls"]:
        if "empty" in wall and wall["empty"]:
            room_id = wall["id"][len("wall|") :]
            room_id = int(room_id[: room_id.find("|")])

            wall_pos_id = wall["id"][len("wall|") :]
            wall_pos_id = wall_pos_id[wall_pos_id.find("|") + 1 :]
            open_wall_connections[wall_pos_id].add(room_id)

    connection_pairs = [set([r1, r2]) for r1, r2 in open_wall_connections.values()]
    connections = []
    while connection_pairs:
        connection = connection_pairs[0]
        offset = 0
        for i, oc in enumerate(connection_pairs[1:].copy()):
            r1, r2 = list(oc)
            if r1 in connection or r2 in connection:
                connection.update(oc)
                del connection_pairs[i - offset]
                offset += 1
        del connection_pairs[0]
        connections.append(connection)

    # Maps the key in the graph to the set of room ids that are assigned to it
    open_room_connection_keys = {
        -(i + 1): connection for i, connection in enumerate(connections)
    }

    # Maps the room id to the key its open neighbor group is assigned in the graph
    room_id_to_open_key = {
        room_id: key
        for key, room_ids in open_room_connection_keys.items()
        for room_id in room_ids
    }

    room_neighbors = defaultdict(set)
    for door in house["doors"]:
        door_room_0 = int(door["room0"].split("|")[-1])
        door_room_1 = int(door["room1"].split("|")[-1])
        if door_room_0 == door_room_1:
            continue
        if door_room_0 in room_id_to_open_key:
            door_room_0 = room_id_to_open_key[door_room_0]
        if door_room_1 in room_id_to_open_key:
            door_room_1 = room_id_to_open_key[door_room_1]
        room_neighbors[door_room_0].add(door_room_1)
        room_neighbors[door_room_1].add(door_room_0)

    graph = nx.Graph()
    for node, neighbors in room_neighbors.items():
        graph.add_node(node)
        for neighbor in neighbors:
            if neighbor not in graph:
                graph.add_node(neighbor)
            graph.add_edge(node, neighbor)

    return graph, room_id_to_open_key


def nearest_room_to_point(point: Dict[str, Any], room_polygons: List[Polygon]) -> int:
    """Get the nearest room from a point."""
    min_dist = float("inf")
    min_room = None
    for room_i, room_poly in enumerate(room_polygons):
        dist = room_poly.exterior.distance(Point(point["x"], point["z"]))
        if dist < min_dist:
            min_dist = dist
            min_room = room_i
    return min_room


def get_approx_geo_dist(
    target_object_type: str,
    agent_position: Dict[str, float],
    house: Dict[str, Any],
    controller: Controller,
    room_polygons: List[Polygon],
    room_connection_graph: nx.Graph,
    room_id_to_open_key: Dict[int, int],
    room_id_with_agent: int,
    house_name: str,
) -> float:
    """Approximates the geodesic distance between the agent and the target object.

    Uses the distance between rooms.
    """
    target_objs = [
        obj
        for obj in controller.last_event.metadata["objects"]
        if obj["objectType"] == target_object_type
    ]
    assert len(target_objs) >= 1

    min_dist = float("inf")
    best_path = []
    for target_obj in target_objs:
        room_i_with_obj = nearest_room_to_point(
            point=target_obj["axisAlignedBoundingBox"]["center"],
            room_polygons=room_polygons,
        )
        room_id_with_obj = int(house["rooms"][room_i_with_obj]["id"].split("|")[-1])

        if room_id_with_obj in room_id_to_open_key:
            room_id_with_obj = room_id_to_open_key[room_id_with_obj]

        path = nx.shortest_path(
            room_connection_graph, room_id_with_agent, room_id_with_obj
        )

        door_connections = {}
        for door in house["doors"]:
            door_room_0 = int(door["room0"].split("|")[-1])
            door_room_1 = int(door["room1"].split("|")[-1])
            if door_room_0 == door_room_1:
                continue

            if door_room_0 in room_id_to_open_key:
                door_room_0 = room_id_to_open_key[door_room_0]
            if door_room_1 in room_id_to_open_key:
                door_room_1 = room_id_to_open_key[door_room_1]

            d1 = min(door_room_0, door_room_1)
            d2 = max(door_room_0, door_room_1)
            door_connections[door["id"]] = (d1, d2)

        door_centers = {}
        for obj in controller.last_event.metadata["objects"]:
            if obj["objectId"] in door_connections:
                door_centers[door_connections[obj["objectId"]]] = {
                    "x": obj["axisAlignedBoundingBox"]["center"]["x"],
                    "z": obj["axisAlignedBoundingBox"]["center"]["z"],
                }
        assert len(door_centers) == len(
            door_connections
        ), "Make sure all doors are enabled in the object metadata SetObjectsFilter"

        dist = 0
        last_position = agent_position
        positions = [agent_position]
        for room0, room1 in zip(path, path[1:]):
            d1 = min(room0, room1)
            d2 = max(room0, room1)
            next_position = door_centers[(d1, d2)]
            positions.append(next_position)
            dist += position_dist(last_position, next_position, ignore_y=True)
            last_position = next_position

        # get the nearest visibility point to the object (of a sample of 10)
        obj_vis_points = controller.step(
            action="GetVisibilityPoints", objectId=target_obj["objectId"]
        ).metadata["actionReturn"]

        rand_vis_points = random.sample(
            population=obj_vis_points,
            k=min(len(obj_vis_points), 10),
        )
        nearest_vis_point_dist = float("inf")
        nearest_vis_point = None
        for vis_point in rand_vis_points:
            vis_point_dist = position_dist(last_position, vis_point, ignore_y=True)
            if vis_point_dist < nearest_vis_point_dist:
                nearest_vis_point_dist = vis_point_dist
                nearest_vis_point = vis_point
        dist += nearest_vis_point_dist
        positions.append(nearest_vis_point)

        if dist < min_dist:
            min_dist = dist
            best_path = [dict(x=p["x"], y=0.95, z=p["z"]) for p in positions]
    return min_dist


def distance_to_object_id(
    controller: Controller,
    distance_cache: DynamicDistanceCache,
    object_id: str,
    house_name: str,
) -> Optional[float]:
    """Minimal geodesic distance to object of given objectId from agent's
    current location.
    It might return -1.0 for unreachable targets.

    # TODO: return None for unreachable targets.
    """

    def path_from_point_to_object_id(
        point: Dict[str, float], object_id: str, allowed_error: float
    ) -> Optional[List[Dict[str, float]]]:
        event = controller.step(
            action="GetShortestPath",
            objectId=object_id,
            position=point,
            allowedError=allowed_error,
        )
        if event:
            return event.metadata["actionReturn"]["corners"]
        else:
            get_logger().debug(
                f"Failed to find path for {object_id} in {house_name}."
                f' Start point {point}, agent state {event.metadata["agent"]}.'
            )
            return None

    def distance_from_point_to_object_id(
        point: Dict[str, float], object_id: str, allowed_error: float
    ) -> float:
        """Minimal geodesic distance from a point to an object of the given
        type.
        It might return -1.0 for unreachable targets.
        """
        path = path_from_point_to_object_id(point, object_id, allowed_error)
        if path:
            # Because `allowed_error != 0` means that the path returned above might not start
            # at `point`, we explicitly add any offset there is.
            dist = position_dist(p0=point, p1=path[0], ignore_y=True)
            return metrics.path_distance(path) + dist
        return -1.0

    def retry_dist(position: Dict[str, float], object_id: str) -> float:
        allowed_error = 0.05
        debug_log = ""
        d = -1.0
        while allowed_error < 2.5:
            d = distance_from_point_to_object_id(position, object_id, allowed_error)
            if d < 0:
                debug_log = (
                    f"In house {house_name}, could not find a path from {position} to {object_id} with"
                    f" {allowed_error} error tolerance. Increasing this tolerance to"
                    f" {2 * allowed_error} any trying again."
                )
                allowed_error *= 2
            else:
                break
        if d < 0:
            get_logger().warning(
                f"In house {house_name}, could not find a path from {position} to {object_id}"
                f" with {allowed_error} error tolerance. Returning a distance of -1."
            )
        elif debug_log != "":
            get_logger().debug(debug_log)
        return d

    return distance_cache.find_distance(
        scene_name=house_name,
        position=controller.last_event.metadata["agent"]["position"],
        target=object_id,
        native_distance_function=retry_dist,
    )
