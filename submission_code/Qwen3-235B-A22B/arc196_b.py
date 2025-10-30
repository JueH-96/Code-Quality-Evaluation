import sys

MOD = 998244353

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    results = []
    for _ in range(T):
        H = int(input[idx])
        W = int(input[idx+1])
        idx += 2
        grid = []
        for _ in range(H):
            grid.append(input[idx])
            idx += 1
        
        valid = True
        
        # Check rows
        for row in grid:
            cnt = row.count('A')
            if cnt % 2 != 0:
                valid = False
                break
        
        if not valid:
            results.append(0)
            continue
        
        # Check columns
        for j in range(W):
            cnt = 0
            for i in range(H):
                if grid[i][j] == 'A':
                    cnt += 1
            if cnt % 2 != 0:
                valid = False
                break
        
        if valid:
            results.append(2)
        else:
            results.append(0)
    
    for res in results:
        print(res)

if __name__ == '__main__':
    main()