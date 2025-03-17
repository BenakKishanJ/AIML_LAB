# Question 2: Develop a program to perform A Star (*) Search Algorithm
import heapq

class Node:
    def __init__(self, state, parent=None, cost=0, heuristic=0):
        self.state = state
        self.parent = parent
        self.cost = cost  # g(n): cost from start to current node
        self.heuristic = heuristic  # h(n): estimated cost from current node to goal
        self.total_cost = cost + heuristic  # f(n) = g(n) + h(n)

    def __lt__(self, other):
        return self.total_cost < other.total_cost

def astar_search(start_state, goal_state, get_neighbors, heuristic_func):
    """
    A* search algorithm implementation

    Args:
        start_state: Initial state
        goal_state: Target state
        get_neighbors: Function that returns neighbors and costs for a state
        heuristic_func: Function that estimates cost from state to goal

    Returns:
        Path from start to goal as a list of states, or None if no path exists
    """
    # Initialize open and closed sets
    open_set = []
    closed_set = set()

    # Create start node and add to open set
    start_node = Node(start_state, None, 0, heuristic_func(start_state, goal_state))
    heapq.heappush(open_set, start_node)

    # Create a dictionary to keep track of nodes by state
    state_to_node = {start_state: start_node}

    while open_set:
        # Get node with lowest f(n) value
        current_node = heapq.heappop(open_set)

        # If goal is reached, reconstruct and return path
        if current_node.state == goal_state:
            path = []
            while current_node:
                path.append(current_node.state)
                current_node = current_node.parent
            return path[::-1]  # Return reversed path

        # Add current state to closed set
        closed_set.add(current_node.state)

        # Explore neighbors
        for neighbor_state, step_cost in get_neighbors(current_node.state):
            # Skip if neighbor is in closed set
            if neighbor_state in closed_set:
                continue

            # Calculate tentative g score
            tentative_g = current_node.cost + step_cost

            # Check if neighbor is in open set
            if neighbor_state in state_to_node and tentative_g >= state_to_node[neighbor_state].cost:
                continue  # This is not a better path

            # Create new node or update existing one
            h = heuristic_func(neighbor_state, goal_state)
            neighbor_node = Node(neighbor_state, current_node, tentative_g, h)
            state_to_node[neighbor_state] = neighbor_node

            # Add to open set
            heapq.heappush(open_set, neighbor_node)

    # No path found
    return None

# Example usage with grid-based pathfinding
def grid_example():
    # Define a simple grid: 0 = open, 1 = obstacle
    grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]

    # Define start and goal positions (row, col)
    start = (0, 0)
    goal = (4, 4)

    # Manhattan distance heuristic
    def heuristic(state, goal):
        return abs(state[0] - goal[0]) + abs(state[1] - goal[1])

    # Get valid neighbors for a position
    def get_neighbors(state):
        row, col = state
        neighbors = []
        # Check all four directions
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            r, c = row + dr, col + dc
            # Check if within grid bounds and not an obstacle
            if (0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0):
                neighbors.append(((r, c), 1))  # Cost of 1 for each step
        return neighbors

    # Run A* search
    path = astar_search(start, goal, get_neighbors, heuristic)

    # Print the result
    print("A* Path from", start, "to", goal, ":")
    if path:
        for i, state in enumerate(path):
            print(f"Step {i}: {state}")

        # Visualize the path on the grid
        path_grid = [row[:] for row in grid]  # Create a copy of the grid
        for r, c in path:
            path_grid[r][c] = 2  # Mark path with 2

        print("\nPath visualization (0=open, 1=obstacle, 2=path):")
        for row in path_grid:
            print(' '.join(str(cell) for cell in row))
    else:
        print("No path found!")

if __name__ == "__main__":
    grid_example()
