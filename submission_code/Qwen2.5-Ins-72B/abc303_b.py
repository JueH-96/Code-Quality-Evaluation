# YOUR CODE HERE
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    photos = [list(map(int, data[2 + i * N: 2 + (i + 1) * N])) for i in range(M)]
    
    # Initialize adjacency matrix to track neighbors
    neighbors = [[False] * N for _ in range(N)]
    
    # Mark pairs that are neighbors in any photo
    for photo in photos:
        for i in range(N - 1):
            a, b = photo[i] - 1, photo[i + 1] - 1
            neighbors[a][b] = True
            neighbors[b][a] = True
    
    # Count pairs that are not neighbors in any photo
    bad_mood_pairs = 0
    for i in range(N):
        for j in range(i + 1, N):
            if not neighbors[i][j]:
                bad_mood_pairs += 1
    
    print(bad_mood_pairs)

if __name__ == "__main__":
    main()