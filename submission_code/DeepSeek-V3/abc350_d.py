import sys
from collections import defaultdict, deque

def main():
    N, M = map(int, sys.stdin.readline().split())
    adj = defaultdict(set)
    for _ in range(M):
        A, B = map(int, sys.stdin.readline().split())
        adj[A].add(B)
        adj[B].add(A)
    
    # To count the number of possible operations
    # We need to find all pairs (X, Z) such that there exists a Y where X and Y are friends, Y and Z are friends, but X and Z are not friends
    # The total number of such pairs is the sum over all Y of (degree(Y) choose 2) minus the number of existing edges
    
    # First, calculate the total possible pairs if all possible connections were made
    total_possible = 0
    for y in adj:
        k = len(adj[y])
        if k >= 2:
            total_possible += k * (k - 1) // 2
    
    # Now, subtract the number of existing edges
    # Since each edge is counted twice (once for each endpoint), we need to divide by 2
    existing_edges = M
    total_possible -= existing_edges
    
    print(total_possible)

if __name__ == "__main__":
    main()