import sys
from collections import defaultdict, deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = defaultdict(list)
    for _ in range(M):
        u, v = map(int, sys.stdin.readline().split())
        adj[u].append(v)
        adj[v].append(u)
    W = list(map(int, sys.stdin.readline().split()))
    A = list(map(int, sys.stdin.readline().split()))
    
    # Initialize the total operations
    total_ops = 0
    # We need to process the vertices in a specific order
    # Since the operation depends on the weight, we should process vertices with higher W first
    # So we sort the vertices in descending order of W
    vertices = sorted(range(1, N+1), key=lambda x: -W[x-1])
    
    for v in vertices:
        if A[v-1] == 0:
            continue
        # For each piece on v, we can perform one operation
        total_ops += A[v-1]
        # Now, for each operation, we need to distribute pieces to adjacent vertices
        # The sum of W_y for the selected S must be less than W_v
        # To maximize the number of operations, we should minimize the number of pieces added
        # So we select the smallest possible S that satisfies the condition
        # Since the sum must be less than W_v, we can select all adjacent vertices whose W_y < W_v
        # But to minimize the number of pieces added, we can select the smallest subset that satisfies the condition
        # However, since the order of processing is from highest W to lowest, we can safely add all adjacent vertices with W_y < W_v
        # Because those vertices will be processed later and their pieces will be handled
        for neighbor in adj[v]:
            if W[neighbor-1] < W[v-1]:
                A[neighbor-1] += A[v-1]
        A[v-1] = 0
    
    print(total_ops)

if __name__ == "__main__":
    main()