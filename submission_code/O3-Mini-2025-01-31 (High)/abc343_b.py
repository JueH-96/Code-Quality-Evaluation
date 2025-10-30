def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    
    n = int(input_data[0])
    idx = 1
    matrix = []
    
    for _ in range(n):
        row = [int(input_data[idx + j]) for j in range(n)]
        matrix.append(row)
        idx += n
    
    for i in range(n):
        # For vertex i, gather vertices j (1-indexed)
        neighbors = [str(j + 1) for j in range(n) if matrix[i][j] == 1]
        print(" ".join(neighbors))
        
if __name__ == '__main__':
    main()