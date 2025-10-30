def find_triplet(N, X, A):
    # Create a dictionary to store the indices of the elements
    index_map = {}
    
    # Store the elements and their indices
    for i in range(N):
        if A[i] not in index_map:
            index_map[A[i]] = []
        index_map[A[i]].append(i + 1)  # Store 1-based index

    # Iterate through pairs (A[i], A[j]) and check for the third element
    for i in range(N):
        for j in range(i + 1, N):
            required = X - (A[i] + A[j])
            if required in index_map:
                # Check for a valid index k
                for k in index_map[required]:
                    if k > j + 1:  # k must be greater than j (1-based index)
                        return (i + 1, j + 1, k)  # Return 1-based indices

    return -1  # No valid triplet found

import sys
input = sys.stdin.read

def main():
    data = input().strip().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:N+2]))

    result = find_triplet(N, X, A)
    if result == -1:
        print(-1)
    else:
        print(*result)

if __name__ == "__main__":
    main()