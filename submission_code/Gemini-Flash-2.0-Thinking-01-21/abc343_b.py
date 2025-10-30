def solve():
    n = int(input())
    adj_matrix = []
    for _ in range(n):
        adj_matrix.append(list(map(int, input().split())))

    for i in range(n):
        neighbors = []
        for j in range(n):
            if adj_matrix[i][j] == 1:
                neighbors.append(j + 1)
        neighbors.sort()
        print(*(neighbors))

if __name__ == "__main__":
    solve()