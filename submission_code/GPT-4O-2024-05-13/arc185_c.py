# YOUR CODE HERE
import sys
input = sys.stdin.read

def find_triple(N, X, A):
    # Dictionary to store pairs and their sums
    pair_sums = {}
    
    # Iterate over each pair (i, j)
    for i in range(N):
        for j in range(i + 1, N):
            pair_sum = A[i] + A[j]
            if pair_sum in pair_sums:
                pair_sums[pair_sum].append((i, j))
            else:
                pair_sums[pair_sum] = [(i, j)]
    
    # Iterate over each element k
    for k in range(N):
        target = X - A[k]
        if target in pair_sums:
            for (i, j) in pair_sums[target]:
                if i < j < k:
                    return i + 1, j + 1, k + 1
    
    return -1

def main():
    data = input().strip().split()
    N = int(data[0])
    X = int(data[1])
    A = list(map(int, data[2:]))
    
    result = find_triple(N, X, A)
    if result == -1:
        print(-1)
    else:
        print(result[0], result[1], result[2])

if __name__ == "__main__":
    main()