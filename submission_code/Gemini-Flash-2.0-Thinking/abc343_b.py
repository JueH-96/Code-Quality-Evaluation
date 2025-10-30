def solve():
    n = int(input())
    adj_matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        adj_matrix.append(row)

    for i in range(n):
        connected_vertices = []
        for j in range(n):
            if adj_matrix[i][j] == 1:
                connected_vertices.append(j + 1)
        connected_vertices.sort()
        print(*connected_vertices)

if __name__ == "__main__":
    solve()