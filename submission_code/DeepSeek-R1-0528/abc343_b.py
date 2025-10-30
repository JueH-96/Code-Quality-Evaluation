def main():
    n = int(input().strip())
    matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)
    
    for i in range(n):
        neighbors = []
        for j in range(n):
            if matrix[i][j] == 1:
                neighbors.append(j + 1)
        neighbors.sort()
        print(' '.join(map(str, neighbors)))

if __name__ == "__main__":
    main()