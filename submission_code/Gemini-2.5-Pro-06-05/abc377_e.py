# YOUR CODE HERE
import sys

def solve():
    """
    Reads input, solves the permutation problem, and prints the result.
    """
    # Use fast I/O
    input = sys.stdin.readline

    # Read N and K from the first line.
    try:
        N, K = map(int, input().split())
    except ValueError:
        return # Exit if input is empty or malformed

    # Read the permutation P from the second line.
    # Convert the 1-based input to a 0-based list for easier indexing.
    # p_map[i] = P_{i+1} - 1
    p_map = [int(x) - 1 for x in input().split()]

    # `visited` array to keep track of elements that have been assigned to a cycle.
    visited = [False] * N
    # `final_p` will store the final permutation after K operations.
    final_p = [0] * N

    # Decompose the permutation into disjoint cycles.
    for i in range(N):
        if not visited[i]:
            # If i has not been visited, it's part of a new cycle.
            # Trace the cycle starting from i.
            current_cycle = []
            curr = i
            while not visited[curr]:
                visited[curr] = True
                current_cycle.append(curr)
                curr = p_map[curr]
            
            cycle_len = len(current_cycle)

            # The operation P_new = P_old o P_old, when applied K times,
            # results in composing the initial permutation P with itself 2^K times.
            # For a cycle of length `cycle_len`, this is equivalent to advancing
            # each element `2^K mod cycle_len` steps along the cycle.
            steps = pow(2, K, cycle_len)

            # Determine the final value for each element in the current cycle.
            for j in range(cycle_len):
                # The element we are considering.
                start_pos = current_cycle[j]
                
                # Its new position in the cycle list after `steps` advancements.
                final_val_index = (j + steps) % cycle_len
                
                # The value it maps to.
                final_val = current_cycle[final_val_index]
                
                # In the final permutation, the value at `start_pos` is `final_val`.
                final_p[start_pos] = final_val

    # Convert the 0-based result back to 1-based and print.
    # The '*' unpacks the generator expression into arguments for print.
    print(*(x + 1 for x in final_p))

solve()