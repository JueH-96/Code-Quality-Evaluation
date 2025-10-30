def solve():
    n = int(input())
    adj_matrix = []
    for _ in range(n):
        row = list(map(int, input().split()))
        adj_matrix.append(row)
    
    for i in range(n):
        neighbors = []
        for j in range(n):
            if adj_matrix[i][j] == 1:
                neighbors.append(j + 1)
        
        neighbors.sort()
        print(*neighbors)

solve()