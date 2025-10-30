import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    adjacency_matrix = []
    index = 1
    for i in range(N):
        adjacency_matrix.append([int(data[index + j]) for j in range(N)])
        index += N
    
    for i in range(N):
        connected_vertices = []
        for j in range(N):
            if adjacency_matrix[i][j] == 1:
                connected_vertices.append(j + 1)
        if connected_vertices:
            print(" ".join(map(str, connected_vertices)))
        else:
            print()

if __name__ == "__main__":
    main()