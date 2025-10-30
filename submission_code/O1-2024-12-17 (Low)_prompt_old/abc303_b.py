def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    idx = 2
    
    # adjacency[i][j] = True if i and j ever stand next to each other in any photo
    adjacency = [[False]*(N+1) for _ in range(N+1)]
    
    for _ in range(M):
        line = list(map(int, data[idx:idx+N]))
        idx += N
        
        # Mark consecutive pairs as adjacent
        for j in range(N - 1):
            x = line[j]
            y = line[j + 1]
            adjacency[x][y] = True
            adjacency[y][x] = True
    
    # Count pairs that never stood next to each other
    answer = 0
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            if not adjacency[i][j]:
                answer += 1
    
    print(answer)

def main():
    solve()

# Call main (solve) here if you need to run the function directly
# But per the problem statement's requirement, we only define solve and call it
# solve()