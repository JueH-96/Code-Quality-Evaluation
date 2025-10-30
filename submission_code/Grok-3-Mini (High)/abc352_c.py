import sys
data = sys.stdin.read().split()
N = int(data[0])
sum_A = 0
max_diff = -1  # Initialize to -1, since B - A >= 0
for i in range(N):
    A = int(data[2 * i + 1])
    B = int(data[2 * i + 2])
    sum_A += A
    diff = B - A
    if diff > max_diff:
        max_diff = diff
answer = sum_A + max_diff
print(answer)