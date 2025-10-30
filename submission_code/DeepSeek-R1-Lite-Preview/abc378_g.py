def main():
    import sys
    from math import factorial
    sys.setrecursionlimit(10000)

    A, B, M = map(int, sys.stdin.read().split())
    N = A * B - 1

    # Precompute factorial and inverse factorial modulo M
    fact = [1] * (N + 1)
    for i in range(1, N + 1):
        fact[i] = fact[i - 1] * i % M

    # Function to compute modular inverse using Fermat's little theorem
    def modinv(x):
        return pow(x, M - 2, M)

    # Function to generate all partitions of 'n' into 'k' parts, each at most 'max_part'
    def generate_partitions(n, k, max_part):
        if k == 1:
            if n <= max_part:
                return [[n]]
            else:
                return []
        partitions = []
        for i in range(min(n, max_part), 0, -1):
            for p in generate_partitions(n - i, k - 1, i):
                partitions.append([i] + p)
        return partitions

    # Compute hook lengths for a given partition
    def compute_hook_lengths(partition):
        # Convert partition to Young diagram coordinates
        # Cells are represented as (row, column)
        # hook_length[r][c] = 1 + (number of cells to the right in the same row) + (number of cells below in the same column)
        rows = len(partition)
        cols = partition[0]
        hook = [[0 for _ in range(partition[r])] for r in range(rows)]
        # Compute hook lengths using formula
        for r in range(rows):
            for c in range(partition[r]):
                hook[r][c] = 1
                # Right cells in the same row
                hook[r][c] += partition[r] - c - 1
                # Cells below in the same column
                for dr in range(r + 1, rows):
                    if c < partition[dr]:
                        hook[r][c] += 1
        return hook

    # Compute f(λ) using hook-length formula
    def f_lambda(partition, fact, M):
        # Compute product of hook lengths
        hook = compute_hook_lengths(partition)
        product = 1
        for r in range(len(hook)):
            for c in range(len(hook[r])):
                product = product * hook[r][c] % M
        # f(λ) = N! / product of hook lengths mod M
        f = fact[N] * modinv(product) % M
        return f

    # Generate all partitions of N - A into B - 1 parts, each at most A - 1
    if B - 1 < 1 or A - 1 < 1:
        print(0)
        return

    target_sum = N - A
    k_parts = B - 1
    max_part = A - 1
    if target_sum < k_parts or target_sum > max_part * k_parts:
        print(0)
        return

    partitions = generate_partitions(target_sum, k_parts, max_part)
    # Add the first part A to each partition
    partitions_with_A = [[A] + p for p in partitions]
    # Now compute f(λ) for each partition and sum them up
    total = 0
    for p in partitions_with_A:
        # Sort parts in non-increasing order
        p_sorted = sorted(p, reverse=True)
        # Compute f(λ)
        f = f_lambda(p_sorted, fact, M)
        total = (total + f) % M
    # Multiply by 2 to account for the additional condition
    total = (total * 2) % M
    print(total)

if __name__ == '__main__':
    main()