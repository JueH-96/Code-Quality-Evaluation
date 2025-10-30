# YOUR CODE HERE
import sys
from itertools import permutations

def main():
    import sys
    import math
    from itertools import permutations

    input = sys.stdin.read().split()
    N = int(input[0])
    K = int(input[1])

    # Generate the multiset of numbers
    multiset = []
    for i in range(1, N + 1):
        multiset.extend([i] * K)

    # Generate all unique permutations
    unique_permutations = sorted(set(permutations(multiset)))

    # Calculate the index of the sequence we need
    index = (len(unique_permutations) + 1) // 2 - 1

    # Get the sequence at the calculated index
    result_sequence = unique_permutations[index]

    # Print the result sequence
    print(" ".join(map(str, result_sequence)))

if __name__ == "__main__":
    main()