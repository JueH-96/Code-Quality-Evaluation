n, t = map(int, input().split())
s = input()
x = list(map(int, input().split()))

# Separate ants by direction
right_ants = []
left_ants = []
for i in range(n):
    if s[i] == '1':
        right_ants.append(x[i])
    else:
        left_ants.append(x[i])

# Sort both lists
right_ants.sort()
left_ants.sort()

count = 0

# For each right-moving ant, count left-moving ants it will meet
import bisect
for r in right_ants:
    # Find left-moving ants in range [r + 1, r + 2 * t]
    left_idx = bisect.bisect_left(left_ants, r + 1)
    right_idx = bisect.bisect_right(left_ants, r + 2 * t)
    count += right_idx - left_idx

print(count)