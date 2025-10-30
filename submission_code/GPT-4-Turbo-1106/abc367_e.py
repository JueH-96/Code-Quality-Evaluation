import sys

def read_ints():
    return list(map(int, sys.stdin.readline().strip().split()))

def apply_operation(A, X, K):
    N = len(A)
    if K == 0:
        return A
    # Find the cycle in the permutation to optimize the number of operations
    visited = [False] * N
    result = [0] * N
    for i in range(N):
        if not visited[i]:
            cycle = []
            current = i
            while not visited[current]:
                visited[current] = True
                cycle.append(current)
                current = X[current] - 1
            cycle_length = len(cycle)
            for j in range(cycle_length):
                result[cycle[j]] = A[cycle[(j - K % cycle_length) % cycle_length]]
    return result

def main():
    N, K = read_ints()
    X = read_ints()
    A = read_ints()
    result = apply_operation(A, X, K)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()