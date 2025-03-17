#Question 1: Develop a program to implement DFS and BFS Search Algorithm.
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def add_edge(self, u, v):
        self.graph[u].append(v)

    def dfs_util(self, v, visited):
        # Mark the current node as visited and print it
        visited.add(v)
        print(v, end=' ')

        # Recur for all the vertices adjacent to this vertex
        for neighbor in self.graph[v]:
            if neighbor not in visited:
                self.dfs_util(neighbor, visited)

    def dfs(self, start):
        # Create a set to store visited vertices
        visited = set()

        print("DFS traversal starting from vertex", start, ":")
        self.dfs_util(start, visited)
        print()

    def bfs(self, start):
        # Create a queue for BFS
        queue = deque([start])

        # Mark the source node as visited
        visited = set([start])

        print("BFS traversal starting from vertex", start, ":")

        while queue:
            # Dequeue a vertex from queue and print it
            vertex = queue.popleft()
            print(vertex, end=' ')

            # Get all adjacent vertices of the dequeued vertex
            # If an adjacent has not been visited, mark it visited and enqueue it
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        print()

# Driver code
if __name__ == '__main__':
    # Create a graph
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)

    # Perform DFS and BFS traversals
    g.dfs(2)
    g.bfs(2)
