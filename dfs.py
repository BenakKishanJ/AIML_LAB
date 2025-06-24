def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()

    if start not in graph:
        print(f"Error: Starting node '{start}' not found in graph.")
        return visited

    visited.add(start)
    print(start)

    # Sort the neighbors to ensure consistent order
    neighbors = sorted(list(graph[start] - visited))
    for next_node in neighbors:
        dfs(graph, next_node, visited)
    return visited


graph = {
    "0": set(["1", "2"]),
    "1": set(["0", "3", "4"]),
    "2": set(["0"]),
    "3": set(["1"]),
    "4": set(["2", "3"]),
}
dfs(graph, "0")
