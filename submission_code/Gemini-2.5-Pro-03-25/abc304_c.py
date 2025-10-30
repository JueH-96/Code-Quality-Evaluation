# YOUR CODE HERE
import sys
import collections

# Function to calculate squared Euclidean distance between two points p1 and p2.
# Using squared distance avoids floating point calculations and potential precision issues with sqrt.
# It's sufficient for comparison since sqrt is monotonically increasing.
def dist_sq(p1, p2):
    """
    Calculates the squared Euclidean distance between two points.
    p1 and p2 are expected to be lists or tuples of coordinates (x, y).
    Returns (p1_x - p2_x)^2 + (p1_y - p2_y)^2.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx*dx + dy*dy

def solve():
    """
    Reads input, performs BFS to determine infected individuals, and prints the result.
    """
    # Read N (number of people) and D (infection distance threshold) from standard input
    N, D = map(int, sys.stdin.readline().split())
    
    # Read coordinates for N people
    # Store coordinates in a list, where coords[i] corresponds to person i+1.
    coords = []
    for _ in range(N):
        coords.append(list(map(int, sys.stdin.readline().split())))

    # Calculate D squared once to use in comparisons. This avoids repeated calculations and using sqrt.
    D_squared = D * D

    # Initialize a boolean list 'infected' to keep track of each person's infection status.
    # Initially, all are False (not infected). Use 0-based indexing internally.
    infected = [False] * N
    
    # Queue for Breadth-First Search (BFS). Stores indices of people to process.
    # A deque is used for efficient append and popleft operations.
    q = collections.deque()

    # Person 1 (at index 0 in our 0-based list) is initially infected.
    # Check if N > 0 is technically redundant due to constraints N >= 1, but good practice.
    if N > 0:
        infected[0] = True  # Mark person 1 (index 0) as infected
        q.append(0)       # Add the index of person 1 to the BFS queue

    # Perform BFS starting from person 1.
    # The loop continues as long as there are infected people whose neighbors haven't been checked.
    while q:
        # Get the index 'u' of the next person from the front of the queue.
        u = q.popleft()
        
        # Iterate through all people 'v' to check if person 'u' can infect them.
        for v in range(N):
            # Check infection condition only if person 'v' is not already infected.
            if not infected[v]:
                 # Calculate the squared distance between person 'u' and person 'v'.
                 # Check if this distance is less than or equal to D_squared.
                 if dist_sq(coords[u], coords[v]) <= D_squared:
                    # If person 'v' is within distance D of infected person 'u', person 'v' gets infected.
                    infected[v] = True
                    # Add the newly infected person 'v' to the queue to process their neighbors later.
                    q.append(v)

    # After the BFS completes, the 'infected' list accurately reflects the final infection status.
    # Print the result for each person (1 to N).
    for i in range(N):
        # The i-th line of output corresponds to person i+1.
        if infected[i]:
            print("Yes")
        else:
            print("No")

# Execute the solve function to run the main logic of the program.
solve()