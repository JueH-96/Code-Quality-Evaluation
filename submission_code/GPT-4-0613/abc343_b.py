N = int(input())
A = [list(map(int, input().split())) for _ in range(N)]

for i in range(N):
    connected_vertices = [j+1 for j, x in enumerate(A[i]) if x == 1]
    print(' '.join(map(str, connected_vertices)))