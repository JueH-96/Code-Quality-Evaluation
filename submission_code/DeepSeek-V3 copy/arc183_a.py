import sys
from math import comb
from functools import lru_cache

def main():
    N, K = map(int, sys.stdin.readline().split())
    total = comb(N * K, K)  # Total number of sequences is (NK)! / (K!)^N
    target = (total + 1) // 2  # The target sequence index

    sequence = []
    counts = [K] * N  # counts[i] represents the remaining count of (i+1)

    for _ in range(N * K):
        for num in range(N):
            if counts[num] == 0:
                continue
            # Calculate the number of sequences if we choose num+1 at this position
            # The remaining counts are counts[0], ..., counts[num]-1, ..., counts[N-1]
            # The total number of sequences is (remaining_total)! / (counts[0]! * ... * (counts[num]-1)! * ... * counts[N-1]!)
            # We can precompute the factorial and use the multinomial coefficient
            # To optimize, we can precompute the factorial and use a recursive function with memoization
            # Here, we use a simplified approach for small N and K
            # For larger N and K, a more optimized approach is needed
            # For the purpose of this problem, we assume that N and K are small enough
            # So we can compute the number of sequences directly
            # Compute the remaining counts
            remaining_counts = counts.copy()
            remaining_counts[num] -= 1
            # Compute the total number of sequences with these remaining counts
            # The total is (sum(remaining_counts))! / (product of factorial of remaining_counts)
            # We can compute this using the multinomial coefficient
            # sum_remaining = sum(remaining_counts)
            # multinomial = factorial(sum_remaining) / (product of factorial of remaining_counts)
            # To compute this, we can use the comb function
            # But since comb is for binomial coefficients, we need a different approach
            # For simplicity, we use a recursive function with memoization
            # Here, we use a helper function to compute the number of sequences
            # with the given remaining counts
            # We can use a memoized function to compute the number of sequences
            # with the given remaining counts
            # The function will return the number of sequences that can be formed
            # with the given remaining counts
            # We can then compare this with the target index
            # If the number of sequences is less than the target, we subtract it from the target
            # and move to the next number
            # Otherwise, we choose this number and continue
            # So, we need a function to compute the number of sequences with the given remaining counts
            # We can use a recursive function with memoization
            # Here, we define the function
            @lru_cache(maxsize=None)
            def count_sequences(remaining_counts_tuple):
                if sum(remaining_counts_tuple) == 0:
                    return 1
                total = 0
                for i in range(len(remaining_counts_tuple)):
                    if remaining_counts_tuple[i] > 0:
                        new_counts = list(remaining_counts_tuple)
                        new_counts[i] -= 1
                        total += count_sequences(tuple(new_counts))
                return total
            # Convert the remaining_counts to a tuple for hashing
            remaining_counts_tuple = tuple(remaining_counts)
            # Compute the number of sequences
            num_sequences = count_sequences(remaining_counts_tuple)
            if num_sequences < target:
                target -= num_sequences
            else:
                sequence.append(num + 1)
                counts[num] -= 1
                break
    print(' '.join(map(str, sequence)))

if __name__ == "__main__":
    main()