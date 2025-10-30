from collections import deque

def main():
    import sys
    from sys import stdin
    input = stdin.read().splitlines()
    
    H_W = input[0].split()
    H = int(H_W[0])
    W = int(H_W[1])
    S = [list(input[i]) for i in range(1, H+1)]
    
    sequence = ['s', 'n', 'u', 'k', 'e']
    
    dx = [0, 1, 0, -1]
    dy = [1, 0, -1, 0]
    
    def is_valid(x, y):
        return 0 <= x < H and 0 <= y < W
    
    def bfs():
        queue = deque()
        if S[0][0] == 's':
            queue.append((0, 0, 0))
        visited = [[[False for _ in range(5)] for _ in range(W)] for _ in range(H)]
        visited[0][0][0] = True
        
        while queue:
            x, y, seq_pos = queue.popleft()
            for i in range(4):
                nx, ny = x + dx[i], y + dy[i]
                if is_valid(nx, ny):
                    next_seq_pos = (seq_pos + 1) % 5
                    if S[nx][ny] == sequence[next_seq_pos] and not visited[nx][ny][next_seq_pos]:
                        if nx == H-1 and ny == W-1:
                            return "Yes"
                        visited[nx][ny][next_seq_pos] = True
                        queue.append((nx, ny, next_seq_pos))
        return "No"
    
    print(bfs())

if __name__ == "__main__":
    main()