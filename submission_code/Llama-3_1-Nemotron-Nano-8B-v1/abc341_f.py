import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = [[] for _ in range(N+1)]  # 1-based indexing
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    W = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    
    # Precompute S for each vertex
    S = [[] for _ in range(N+1)]
    for x in range(1, N+1):
        neighbors = adj[x]
        # Sort neighbors by their W in ascending order
        sorted_neighbors = sorted(neighbors, key=lambda y: W[y-1])
        prefix = [0]
        current_sum = 0
        for y in sorted_neighbors:
            current_sum += W[y-1]
            prefix.append(current_sum)
        # Find maximum k where prefix[k] < W[x-1]
        k = 0
        for i in range(1, len(prefix)):
            if prefix[i] < W[x-1]:
                k = i
            else:
                break
        S[x] = sorted_neighbors[:k]
    
    # Sort vertices in descending order of W
    sorted_vertices = sorted(range(1, N+1), key=lambda x: -W[x-1])
    
    total_ops = 0
    for x in sorted_vertices:
        if A[x-1] == 0:
            continue
        total_ops += A[x-1]
        for y in S[x]:
            A[y-1] += A[x-1]
        A[x-1] = 0
    
    print(total_ops)

if __name__ == '__main__':
    main()