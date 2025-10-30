import sys

def is_possible(N, K):
    # Check if it's possible to have exactly K fixed elements in an N x N matrix
    if K == 0:
        return True
    if K == N * N:
        return True
    if K == N:
        return True
    if K == N * (N - 1):
        return True
    return False

def main():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1
    Q = int(data[index])
    index += 1

    queries = []
    for _ in range(Q):
        K = int(data[index])
        index += 1
        queries.append(K)

    results = []
    for K in queries:
        if is_possible(N, K):
            results.append("Yes")
        else:
            results.append("No")

    for result in results:
        print(result)

if __name__ == "__main__":
    main()