import sys
from collections import deque
import bisect

# Function to check if a square x is bad
def is_bad(x, bad_intervals):
    if not bad_intervals:
        return False
    # Find the first interval [L, R] where L > x
    # bisect_right finds insertion point to maintain order,
    # elements to the left are <= x
    i = bisect.bisect_right([pair[0] for pair in bad_intervals], x)
    # If i > 0, check the interval [L_{i-1}, R_{i-1}]
    if i > 0:
        if bad_intervals[i-1][0] <= x <= bad_intervals[i-1][1]:
            return True
    return False

def solve():
    # Read input using sys.stdin.readline
    line1 = sys.stdin.readline().split()
    N = int(line1[0])
    M = int(line1[1])
    A = int(line1[2])
    B = int(line1[3])

    bad_intervals = []
    for _ in range(M):
        L, R = map(int, sys.stdin.readline().split())
        bad_intervals.append((L, R))

    # Build the set of important good squares V_set
    # These are 1, N, and squares L_i - k, R_i + k for 0 <= k <= B that are good and within [1, N]
    V_set = {1}
    if N > 1:
        V_set.add(N)

    for L, R in bad_intervals:
        # Squares before L_i
        for k in range(B + 1):
            pre_square = L - k
            # Only consider squares within [1, N]
            if pre_square >= 1 and pre_square <= N:
                 # Check if the square is good
                if not is_bad(pre_square, bad_intervals):
                    V_set.add(pre_square)
        # Squares after R_i
        for k in range(B + 1):
            post_square = R + k
            # Only consider squares within [1, N]
            if post_square >= 1 and post_square <= N:
                # Check if the square is good
                if not is_bad(post_square, bad_intervals):
                    V_set.add(post_square)

    # Sort the important nodes and create a mapping
    V_list = sorted(list(V_set))
    node_to_idx = {v: i for i, v in enumerate(V_list)}

    # Build the adjacency list for the graph
    # An edge u -> v exists if u, v are in V_set, A <= v - u <= B, and v is good.
    # v is guaranteed good if it is in V_set by construction.
    adj = [[] for _ in range(len(V_list))]

    for i in range(len(V_list)):
        u = V_list[i]
        
        # Find the index of the first node in V_list that is >= u + A
        j_start = bisect.bisect_left(V_list, u + A)

        # Iterate through potential neighbors v in V_list starting from j_start
        for j in range(j_start, len(V_list)):
            v = V_list[j]

            # If v is beyond the maximum jump distance from u, stop checking for this u
            if v > u + B:
                break

            # v is guaranteed to be >= u + A and <= u + B because of the loop bounds and V_list being sorted.
            # v is guaranteed to be in V_list and good by construction.
            
            # Add the edge u -> v
            adj[i].append(j)

    # BFS to check reachability from 1 to N
    # Check if start and end nodes exist in V_list.
    # 1 is always in V_set. N is always in V_set if N >= 2.
    if 1 not in node_to_idx or N not in node_to_idx:
         # This case should ideally not be reachable given constraints and V_set logic
         print("No") # Defensive programming
         return

    start_idx = node_to_idx[1]
    end_idx = node_to_idx[N]

    q = deque([start_idx])
    visited = {start_idx}

    while q:
        curr_idx = q.popleft()

        if curr_idx == end_idx:
            print("Yes")
            return

        # Explore neighbors
        for neighbor_idx in adj[curr_idx]:
            if neighbor_idx not in visited:
                visited.add(neighbor_idx)
                q.append(neighbor_idx)

    # If BFS finishes and N is not reached
    print("No")

solve()