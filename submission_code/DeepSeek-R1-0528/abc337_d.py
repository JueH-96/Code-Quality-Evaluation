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

    if W >= K:
        for i in range(H):
            count_dot = 0
            count_x = 0
            for j in range(0, K):
                c = grid[i][j]
                if c == '.':
                    count_dot += 1
                elif c == 'x':
                    count_x += 1
            
            if count_x == 0:
                min_ops = min(min_ops, count_dot)
                
            for j in range(0, W-K):
                c_remove = grid[i][j]
                if c_remove == '.':
                    count_dot -= 1
                elif c_remove == 'x':
                    count_x -= 1
                    
                c_add = grid[i][j+K]
                if c_add == '.':
                    count_dot += 1
                elif c_add == 'x':
                    count_x += 1
                    
                if count_x == 0:
                    min_ops = min(min_ops, count_dot)
                    
    if H >= K:
        for j in range(W):
            count_dot = 0
            count_x = 0
            for i in range(0, K):
                c = grid[i][j]
                if c == '.':
                    count_dot += 1
                elif c == 'x':
                    count_x += 1
                    
            if count_x == 0:
                min_ops = min(min_ops, count_dot)
                
            for i in range(0, H-K):
                c_remove = grid[i][j]
                if c_remove == '.':
                    count_dot -= 1
                elif c_remove == 'x':
                    count_x -= 1
                    
                c_add = grid[i+K][j]
                if c_add == '.':
                    count_dot += 1
                elif c_add == 'x':
                    count_x += 1
                    
                if count_x == 0:
                    min_ops = min(min_ops, count_dot)
                    
    print(min_ops if min_ops != INF else -1)

if __name__ == "__main__":
    main()