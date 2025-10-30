def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    adj_matrix = []
    for i in range(1, n + 1):
        row = list(map(int, data[i].split()))
        adj_matrix.append(row)
    
    output_lines = []
    for i in range(n):
        neighbors = []
        for j in range(n):
            if adj_matrix[i][j] == 1:
                neighbors.append(str(j + 1))
        output_lines.append(" ".join(neighbors))
    
    print("
".join(output_lines))

if __name__ == "__main__":
    main()