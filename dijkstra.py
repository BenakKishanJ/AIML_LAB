# Dijkstra's algorithm to find the shortest path

def dijkstra(n, cost, src, s, d, p):
    s[src] = 1
    for i in range(n):
        u = -1
        min_val = 999  # Placeholder for infinity

        for j in range(n):
            if s[j] == 0 and d[j] < min_val:
                min_val = d[j]
                u = j

        if u == -1:  # No reachable node left
            return

        s[u] = 1

        for v in range(n):
            if s[v] == 0 and cost[u][v] != 999 and d[u] + cost[u][v] < d[v]:
                d[v] = d[u] + cost[u][v]
                p[v] = u

def print_path(n, src, dest, d, p):
    if d[dest] >= 999:
        print(f"No path exists from {src} to {dest}!\n")
        return

    path = []
    i = dest
    while i != src:
        path.append(i)
        i = p[i]
    path.append(src)

    print(" -> ".join(map(str, path[::-1])), f"= {d[dest]}")


def main():
    n = int(input("Enter the number of nodes: "))

    # Initialize cost matrix with "infinity" (999) for unreachable nodes
    cost = [[999] * n for _ in range(n)]

    print("Enter the adjacency cost matrix (use 999 for no direct path):")
    for i in range(n):
        for j in range(n):
            cost[i][j] = int(input(f"cost[{i}][{j}]: "))

    src = int(input("Enter the source node (0-based index): "))

    # Initialize distance array as the cost from the source node
    d = cost[src][:]  # Copy the source row from the cost matrix
    s = [0] * n       # Visited nodes
    p = [src] * n      # Parent array for path tracking

    d[src] = 0  # Distance to itself is 0

    dijkstra(n, cost, src, s, d, p)

    print("\nShortest paths from source to all other nodes:\n")
    for i in range(n):
        if i != src:
            print_path(n, src, i, d, p)

if __name__ == "__main__":
    main()
