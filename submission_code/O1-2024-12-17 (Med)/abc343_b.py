def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    adj_matrix = []
    index = 1
    for _ in range(N):
        row = list(map(int, data[index:index+N]))
        index += N
        adj_matrix.append(row)
    
    for i in range(N):
        neighbors = [str(j+1) for j in range(N) if adj_matrix[i][j] == 1]
        print(" ".join(neighbors))

if __name__ == "__main__":
    main()