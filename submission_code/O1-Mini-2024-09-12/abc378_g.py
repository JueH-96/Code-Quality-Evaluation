# YOUR CODE HERE
import sys
import math
from functools import lru_cache

def main():
    import sys
    sys.setrecursionlimit(10000)
    A, B, M = map(int, sys.stdin.read().split())
    N = A * B - 1

    # Precompute factorial modulo M
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = (fact[i - 1] * i) % M

    # Function to compute all partitions with exactly B parts, first part A
    def generate_partitions(A, B, remaining, current_partition):
        if B == 1:
            if 1 <= remaining <= A:
                yield current_partition + [remaining]
            return
        for next_part in range(1, min(A, remaining - (B -1)) +1):
            if current_partition:
                if next_part > current_partition[-1]:
                    continue
            yield from generate_partitions(A, B -1, remaining - next_part, current_partition + [next_part])

    # Function to compute hook lengths for a given partition
    def compute_hook_lengths(partition):
        hook_lengths = []
        # Number of rows
        rows = len(partition)
        # Number of columns
        cols = max(partition)
        for i in range(rows):
            for j in range(partition[i]):
                # Count cells to the right in the same row
                right = partition[i] - j -1
                # Count cells below in the same column
                below = 0
                for k in range(i+1, rows):
                    if partition[k] > j:
                        below +=1
                    else:
                        break
                hook_lengths.append(right + below +1)
        return hook_lengths

    # Generate all valid partitions
    partitions = list(generate_partitions(A, B, N - A, [A]))
    # Verify lambda'_1 = B
    valid_partitions = []
    for partition in partitions:
        # Compute the conjugate partition
        conjugate = []
        for i in range(B):
            count = 0
            for part in partition:
                if part > i:
                    count +=1
            conjugate.append(count)
        if conjugate[0] == B:
            valid_partitions.append(partition)

    total_p = 0
    total_p_end_with_max = 0

    for partition in valid_partitions:
        hook_lengths = compute_hook_lengths(partition)
        prod_hooks = 1
        for h in hook_lengths:
            prod_hooks = (prod_hooks * h) % M
        inv_prod_hooks = pow(prod_hooks, M -2, M)
        f_lambda = (fact[N] * inv_prod_hooks) % M
        total_p = (total_p + f_lambda * f_lambda) % M
        total_p_end_with_max = (total_p_end_with_max + A * f_lambda) % M

    answer = (total_p - total_p_end_with_max) % M
    print(answer)

if __name__ == "__main__":
    main()