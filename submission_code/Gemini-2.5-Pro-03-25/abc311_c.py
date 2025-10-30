# YOUR CODE HERE
import sys

# By default, recursion depth limit in Python is relatively low.
# While this solution is iterative, Python's internals might involve deep calls for large structures.
# Setting a higher limit ensures safety for large N, though likely not strictly necessary for this iterative code.
# It is safer to set it for competitive programming problems involving potentially deep paths/structures.
sys.setrecursionlimit(2 * 10**5 + 10) 

def solve():
    # Read the number of vertices N
    N = int(sys.stdin.readline())
    
    # Read the N integers A_1, ..., A_N which define the edges.
    # A[i] corresponds to the destination vertex for the edge starting from vertex i+1 (using 1-based indexing).
    A = list(map(int, sys.stdin.readline().split()))

    # Convert the input array A to an adjacency list representation using 0-based indexing.
    # adj[i] will store the 0-based index of the vertex that vertex i (0-based) points to.
    # Since each vertex has exactly one outgoing edge, adj[i] stores that single destination vertex index.
    # The input edge is from vertex i+1 to vertex A[i]. In 0-based indexing, this is an edge from vertex i to vertex A[i]-1.
    adj = [0] * N
    for i in range(N):
        adj[i] = A[i] - 1

    # visited_step[i] will store the step number (or "time") at which vertex i (0-based index) was first visited during the traversal.
    # Initialize all entries to -1, indicating that no vertex has been visited yet.
    visited_step = [-1] * N
    
    # path will store the sequence of vertices (0-based indices) visited in the current traversal path.
    path = []
    
    # Start the traversal from vertex 0 (which corresponds to vertex 1 in the problem statement).
    # Since the problem guarantees a cycle exists, starting from any vertex within a component that contains a cycle
    # will eventually lead to finding that cycle. Starting from vertex 0 is a standard choice.
    curr = 0
    step = 0
    
    # Follow the path defined by the adj array. The loop continues as long as we visit vertices that haven't been visited before in this traversal.
    # The condition `visited_step[curr] == -1` checks if the current vertex `curr` has been visited previously in this path traversal.
    while visited_step[curr] == -1:
        # Mark the current vertex `curr` as visited at the current `step`. Record the step number.
        visited_step[curr] = step
        # Add the current vertex `curr` to the path list to keep track of the traversal sequence.
        path.append(curr)
        
        # Move to the next vertex in the sequence by following the unique outgoing edge from `curr`.
        curr = adj[curr]
        # Increment the step counter.
        step += 1

    # If the loop terminates, it means `visited_step[curr]` was not -1.
    # This indicates that the vertex `curr` has been visited before during this traversal.
    # We have found a cycle. The vertex `curr` is the first vertex encountered that closes the cycle.
    
    # Find the step number when `curr` was first visited. This recorded step number marks the beginning of the cycle within our path list.
    start_step = visited_step[curr]
    
    # The vertices forming the cycle are those in the path list starting from index `start_step` up to the end of the path recorded so far.
    # This slice `path[start_step:]` captures the sequence of vertices: path[start_step], path[start_step+1], ..., path[step-1].
    # These vertices form the simple directed cycle.
    cycle = path[start_step:]
    
    # Output the length of the cycle found (M).
    print(len(cycle))
    
    # Output the vertices in the cycle (B_1, B_2, ..., B_M).
    # Convert each vertex index from 0-based back to 1-based indexing by adding 1 before printing.
    # The `*` operator unpacks the generator expression `(v + 1 for v in cycle)` into individual arguments for the print function,
    # resulting in space-separated output.
    print(*(v + 1 for v in cycle))

# Call the solve function to execute the solution logic.
solve()