import sys
from collections import defaultdict

def count_triples(N, A):
    # Dictionary to store the indices of each element
    indices = defaultdict(list)
    for i, value in enumerate(A):
        indices[value].append(i)

    count = 0

    # Iterate over each unique value in the array
    for value, idx_list in indices.items():
        # If there are less than 2 occurrences, we can't form a valid triple
        if len(idx_list) < 2:
            continue

        # Iterate over each pair of indices (i, k) where A[i] == A[k]
        for i in range(len(idx_list)):
            for k in range(i + 1, len(idx_list)):
                # Count the number of valid j's such that i < j < k and A[j] != A[i]
                count += (idx_list[k] - idx_list[i] - 1) - (k - i - 1)

    return count

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    A = list(map(int, data[1:]))

    result = count_triples(N, A)
    print(result)

if __name__ == "__main__":
    main()