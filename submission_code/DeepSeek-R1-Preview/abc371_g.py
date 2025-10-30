import sys
from math import gcd

def main():
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    P = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    P = [p - 1 for p in P]  # 0-based indexing
    A = A  # 1-based values

    # Decompose into cycles
    visited = [False] * N
    cycles = []
    for i in range(N):
        if not visited[i]:
            cycle = []
            j = i
            while not visited[j]:
                visited[j] = True
                cycle.append(j)
                j = P[j]
            cycles.append(cycle)
    # Sort cycles by their minimal element
    cycles.sort(key=lambda x: min(x))
    # Get the earliest cycle
    if not cycles:
        print(' '.join(map(str, A)))
        return
    earliest_cycle = cycles[0]
    m = len(earliest_cycle)
    # Generate all possible rotations for the earliest cycle
    # For each rotation, compute the entire array and track the minimal one
    min_array = None
    earliest_sorted = sorted(earliest_cycle)
    for r in range(m):
        # Compute rotated values for earliest_cycle
        rotated = []
        for i in range(len(earliest_sorted)):
            pos = earliest_sorted[i]
            idx_in_cycle = earliest_cycle.index(pos)
            new_idx = (idx_in_cycle + r) % m
            rotated_val = A[earliest_cycle[new_idx]]
            rotated.append(rotated_val)
        # Create a candidate array
        candidate = A.copy()
        for i in range(len(earliest_sorted)):
            pos = earliest_sorted[i]
            candidate[pos] = rotated[i]
        # Now, process other cycles with k = r mod len(cycle)
        for cycle in cycles[1:]:
            cycle_len = len(cycle)
            k = r % cycle_len
            # Sort the cycle's positions to maintain array order
            sorted_cycle = sorted(cycle)
            # Rotate the cycle's values
            rotated_values = []
            for i in range(len(sorted_cycle)):
                pos = sorted_cycle[i]
                idx_in_cycle = cycle.index(pos)
                new_idx = (idx_in_cycle + k) % cycle_len
                rotated_values.append(A[cycle[new_idx]])
            # Update the candidate array
            for i in range(len(sorted_cycle)):
                pos = sorted_cycle[i]
                candidate[pos] = rotated_values[i]
        # Compare with min_array
        if min_array is None or candidate < min_array:
            min_array = candidate.copy()
    # Output min_array
    print(' '.join(map(str, min_array)))

if __name__ == '__main__':
    main()