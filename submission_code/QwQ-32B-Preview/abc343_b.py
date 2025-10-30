N = int(input())
adj_matrix = []
for _ in range(N):
    line = input()
    row = list(map(int, line.split()))
    adj_matrix.append(row)

for i in range(N):
    connected = [str(j+1) for j in range(N) if adj_matrix[i][j] == 1]
    connected_sorted = ' '.join(sorted(connected, key=int))
    print(connected_sorted)