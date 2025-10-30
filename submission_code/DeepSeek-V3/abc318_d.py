# YOUR CODE HERE
import sys
import itertools

def main():
    N = int(sys.stdin.readline())
    D = []
    for i in range(N-1):
        D.append(list(map(int, sys.stdin.readline().split())))
    
    # Create a full adjacency matrix
    adj = [[0]*N for _ in range(N)]
    for i in range(N-1):
        for j in range(i+1, N):
            adj[i][j] = D[i][j-i-1]
            adj[j][i] = D[i][j-i-1]
    
    # Generate all possible matchings
    # Since N is small (up to 16), we can use bitmasking
    # But for N=16, the number of matchings is 16!! which is 2027025, manageable
    # We need to find the maximum weight matching
    
    # To handle the matching, we can use a recursive approach with memoization
    # or use a bitmask to represent the used vertices
    
    # Using a bitmask to represent the used vertices
    # We will try to pair each vertex with another, ensuring no overlaps
    
    # Precompute all possible pairs and their weights
    pairs = []
    for i in range(N):
        for j in range(i+1, N):
            pairs.append((i, j, adj[i][j]))
    
    # Sort pairs by weight in descending order
    pairs.sort(key=lambda x: -x[2])
    
    # Use a greedy approach to select pairs
    used = [False] * N
    total = 0
    for p in pairs:
        i, j, w = p
        if not used[i] and not used[j]:
            used[i] = True
            used[j] = True
            total += w
    print(total)

if __name__ == "__main__":
    main()