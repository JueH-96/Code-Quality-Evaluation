def count_valid_placements(H, W, grid):
    MOD = 998244353
    
    # Count the number of A and B tiles
    count_A = sum(row.count('A') for row in grid)
    count_B = sum(row.count('B') for row in grid)
    
    # Check for dead ends
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'A':
                # Check right neighbor
                if grid[i][(j + 1) % W] == 'B':
                    return 0
                # Check bottom neighbor
                if grid[(i + 1) % H][j] == 'B':
                    return 0
            elif grid[i][j] == 'B':
                # Check right neighbor
                if grid[i][(j + 1) % W] == 'A':
                    return 0
                # Check bottom neighbor
                if grid[(i + 1) % H][j] == 'A':
                    return 0

    # Calculate the number of valid placements
    valid_placements = pow(4, count_A, MOD) * pow(2, count_B, MOD) % MOD
    return valid_placements

def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    index = 0
    T = int(data[index])
    index += 1
    results = []
    
    for _ in range(T):
        H, W = map(int, data[index].split())
        index += 1
        grid = [data[index + i] for i in range(H)]
        index += H
        
        result = count_valid_placements(H, W, grid)
        results.append(result)
    
    print('
'.join(map(str, results)))

if __name__ == "__main__":
    main()