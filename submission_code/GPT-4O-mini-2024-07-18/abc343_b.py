def main():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    adjacency_matrix = [list(map(int, line.split())) for line in data[1:N+1]]
    
    for i in range(N):
        connected_vertices = []
        for j in range(N):
            if adjacency_matrix[i][j] == 1:
                connected_vertices.append(j + 1)  # +1 to convert 0-indexed to 1-indexed
        print(" ".join(map(str, sorted(connected_vertices))))

if __name__ == "__main__":
    main()