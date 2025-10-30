# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

count = 0
for color in range(1, N+1):
    positions = [i for i, x in enumerate(A) if x == color]
    if len(positions) == 2 and positions[1] - positions[0] == 2:
        count += 1

print(count)