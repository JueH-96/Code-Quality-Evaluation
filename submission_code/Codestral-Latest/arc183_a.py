import sys
from itertools import permutations

def find_kth_good_sequence(N, K):
    # Generate all permutations of the sequence [1, 1, ..., 2, 2, ..., N, N]
    sequence = [i for i in range(1, N+1) for _ in range(K)]
    total_sequences = factorial(N * K) // (factorial(K) ** N)

    # Calculate the target index
    target_index = (total_sequences + 1) // 2 - 1

    # Generate all permutations and sort them lexicographically
    all_permutations = sorted(set(permutations(sequence)))

    # Return the target sequence
    return ' '.join(map(str, all_permutations[target_index]))

def factorial(n):
    if n == 0:
        return 1
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    K = int(data[1])

    result = find_kth_good_sequence(N, K)
    print(result)