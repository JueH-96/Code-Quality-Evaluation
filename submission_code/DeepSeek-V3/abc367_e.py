# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = int(data[1])
    X = list(map(int, data[2:2+N]))
    A = list(map(int, data[2+N:2+2*N]))
    
    # Since K can be up to 1e18, we need to find cycles in the permutation
    # We will find the cycle for each position and determine the final value after K steps
    
    # First, we need to find the cycles in the permutation X
    visited = [False] * (N + 1)
    cycles = []
    
    for i in range(1, N+1):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = X[current-1]
            cycles.append(cycle)
    
    # Now, for each cycle, we can compute the final value after K steps
    # Since the cycle length is len(cycle), the effective steps is K % len(cycle)
    
    # Create a mapping from original index to its position in the cycle
    # and the cycle it belongs to
    pos_in_cycle = {}
    cycle_id = {}
    for idx, cycle in enumerate(cycles):
        for pos, val in enumerate(cycle):
            pos_in_cycle[val] = pos
            cycle_id[val] = idx
    
    # Now, for each element in A, we find its cycle and the position in the cycle
    # and compute the final value after K steps
    A_final = [0] * N
    for i in range(N):
        original_index = i + 1
        if K == 0:
            A_final[i] = A[i]
            continue
        # Find the cycle and the position
        cid = cycle_id[original_index]
        cycle = cycles[cid]
        pos = pos_in_cycle[original_index]
        # The effective steps is K % len(cycle)
        steps = K % len(cycle)
        # The new position is (pos + steps) % len(cycle)
        new_pos = (pos + steps) % len(cycle)
        # The new index is cycle[new_pos]
        new_index = cycle[new_pos]
        # The value is A[new_index - 1]
        A_final[i] = A[new_index - 1]
    
    print(' '.join(map(str, A_final)))

if __name__ == "__main__":
    main()