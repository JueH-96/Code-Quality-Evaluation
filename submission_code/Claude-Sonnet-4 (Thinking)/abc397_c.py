n = int(input())
a = list(map(int, input().split()))

# Compute left_distinct
left_distinct = [0] * n
seen = set()
for i in range(n):
    seen.add(a[i])
    left_distinct[i] = len(seen)

# Compute right_distinct
right_distinct = [0] * n
seen = set()
for i in range(n-1, -1, -1):
    seen.add(a[i])
    right_distinct[i] = len(seen)

# Find maximum sum
max_sum = 0
for i in range(1, n):
    current_sum = left_distinct[i-1] + right_distinct[i]
    max_sum = max(max_sum, current_sum)

print(max_sum)