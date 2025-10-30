# YOUR CODE HERE
n = int(input())
total_shoulder_sum = 0
max_diff = -float('inf')

for _ in range(n):
    a, b = map(int, input().split())
    total_shoulder_sum += a
    max_diff = max(max_diff, b - a)

print(total_shoulder_sum + max_diff)