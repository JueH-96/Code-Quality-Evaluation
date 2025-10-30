import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print(-1)
        return
        
    first_line = data[0].split()
    H = int(first_line[0])
    W = int(first_line[1])
    K = int(first_line[2])
    grid = []
    for i in range(1, 1+H):
        grid.append(data[i].strip())
    
    INF = 10**9
    min_ops = INF
    
    if K <= W:
        for i in range(H):
            x_count = 0
            dot_count = 0
            for j in range(K):
                c = grid[i][j]
                if c == 'x':
                    x_count += 1
                elif c == '.':
                    dot_count += 1
                    
            if x_count == 0:
                min_ops = min(min_ops, dot_count)
                
            for j in range(1, W - K + 1):
                left_char = grid[i][j-1]
                if left_char == 'x':
                    x_count -= 1
                elif left_char == '.':
                    dot_count -= 1
                    
                right_char = grid[i][j+K-1]
                if right_char == 'x':
                    x_count += 1
                elif right_char == '.':
                    dot_count += 1
                    
                if x_count == 0:
                    min_ops = min(min_ops, dot_count)
                    
    if K <= H:
        for j in range(W):
            x_count = 0
            dot_count = 0
            for i in range(K):
                c = grid[i][j]
                if c == 'x':
                    x_count += 1
                elif c == '.':
                    dot_count += 1
                    
            if x_count == 0:
                min_ops = min(min_ops, dot_count)
                
            for i in range(1, H - K + 1):
                top_char = grid[i-1][j]
                if top_char == 'x':
                    x_count -= 1
                elif top_char == '.':
                    dot_count -= 1
                    
                bottom_char = grid[i+K-1][j]
                if bottom_char == 'x':
                    x_count += 1
                elif bottom_char == '.':
                    dot_count += 1
                    
                if x_count == 0:
                    min_ops = min(min_ops, dot_count)
                    
    if min_ops == INF:
        print(-1)
    else:
        print(min_ops)

if __name__ == "__main__":
    main()