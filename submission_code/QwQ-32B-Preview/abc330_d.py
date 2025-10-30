def count_triples(N, grid):
    # Count 'o's in each row
    r = [row.count('o') for row in grid]
    # Count 'o's in each column
    c = [0] * N
    for col in range(N):
        for row in range(N):
            if grid[row][col] == 'o':
                c[col] += 1
    # Calculate total triples
    total = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                total += (r[i] - 1) * (c[j] - 1)
    return total

def main():
    import sys
    data = sys.stdin.read().splitlines()
    N = int(data[0])
    grid = data[1:1+N]
    print(count_triples(N, grid))

if __name__ == "__main__":
    main()