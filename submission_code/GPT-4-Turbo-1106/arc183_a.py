from math import factorial
from itertools import permutations

def find_median_sequence(N, K):
    # Calculate the total number of good integer sequences
    total_sequences = factorial(N * K) // (factorial(K) ** N)
    # Find the median index (1-based)
    median_index = (total_sequences + 1) // 2

    # Generate the initial sequence
    sequence = [i for i in range(1, N + 1) for _ in range(K)]
    # Generate all unique permutations of the sequence
    unique_permutations = set(permutations(sequence))
    # Sort the unique permutations lexicographically
    sorted_permutations = sorted(unique_permutations)

    # Find the median sequence (0-based index)
    median_sequence = sorted_permutations[median_index - 1]
    return median_sequence

def main():
    N, K = map(int, input().split())
    median_sequence = find_median_sequence(N, K)
    print(' '.join(map(str, median_sequence)))

if __name__ == "__main__":
    main()