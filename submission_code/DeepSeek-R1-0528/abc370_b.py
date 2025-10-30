def main():
    n = int(input().strip())
    matrix = []
    for i in range(n):
        row = list(map(int, input().split()))
        matrix.append(row)
    
    current = 1
    for j in range(1, n + 1):
        if current >= j:
            current = matrix[current - 1][j - 1]
        else:
            current = matrix[j - 1][current - 1]
            
    print(current)

if __name__ == "__main__":
    main()