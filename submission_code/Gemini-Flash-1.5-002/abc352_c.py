# YOUR CODE HERE
N = int(input())
giants = []
for i in range(N):
    A, B = map(int, input().split())
    giants.append((B - A, A, B))

giants.sort(reverse=True)

max_height = 0
current_height = 0
for i in range(N):
    if i == 0:
        current_height = giants[i][2]
    else:
        current_height += giants[i][1]
    max_height = max(max_height, current_height)

print(max_height)