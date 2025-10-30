def count_triples(grid, N):
    count = 0
    for i in range(N):
        for j in range(N):
            if grid[i][j] == 'o':
                for k in range(N):
                    if k != i and grid[k][j] == 'o' and grid[i][k] == 'o':
                        count += 1
    return count

def main():
    N = int(input().strip())
    grid = [input().strip() for _ in range(N)]
    result = count_triples(grid, N)
    print(result)

if __name__ == "__main__":
    main()