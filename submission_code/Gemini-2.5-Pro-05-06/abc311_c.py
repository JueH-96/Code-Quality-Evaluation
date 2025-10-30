import sys

def solve():
    N = int(sys.stdin.readline())
    # A_values are 1-indexed targets for 1-indexed sources.
    # A_values[k] is the target for source k+1.
    A_values = list(map(int, sys.stdin.readline().split()))

    # adj[v] will store the vertex that v points to.
    # Vertices are 1 to N.
    # adj is 1-indexed for convenience.
    adj = [0] * (N + 1)
    for i in range(N):
        # A_values[i] is the target for vertex i+1
        adj[i+1] = A_values[i]

    # visited_step[v] stores the step number (1-indexed) at which vertex v was first visited
    # in the current traversal. 0 means not visited yet in current traversal.
    visited_step = [0] * (N + 1)
    
    # path_recorder_list will store vertices in the order they are visited.
    # It's 0-indexed. path_recorder_list[k] is the vertex visited at (k+1)-th step.
    path_recorder_list = [] 

    # Start traversal from vertex 1 (any vertex would eventually lead to a cycle).
    # Vertex 1 is guaranteed to exist since N >= 2.
    current_v = 1 
    current_s = 1 # Current step number (1-indexed)

    while True:
        # If current_v has been visited in this traversal
        if visited_step[current_v] != 0:
            # Cycle detected.
            # current_v was first visited at step 'first_occurrence_step'.
            first_occurrence_step = visited_step[current_v]
            
            # The path_recorder_list is 0-indexed.
            # The vertex visited at step S (1-indexed) is at path_recorder_list[S-1].
            # So, the first occurrence of current_v is at index (first_occurrence_step - 1).
            start_idx_in_list = first_occurrence_step - 1
            
            # The cycle consists of vertices from this first occurrence up to
            # the vertex visited just before re-encountering current_v.
            # These are all currently in path_recorder_list from start_idx_in_list onwards.
            cycle = path_recorder_list[start_idx_in_list:] 
            
            print(len(cycle))
            print(*(cycle)) # Unpack list elements as arguments to print
            return
        
        # Mark current_v as visited at current_s
        visited_step[current_v] = current_s
        # Record current_v in the path
        path_recorder_list.append(current_v)
        
        # Move to the next vertex
        current_v = adj[current_v]
        # Increment step number
        current_s += 1

if __name__ == '__main__':
    solve()