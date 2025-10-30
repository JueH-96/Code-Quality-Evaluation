# Read input
N, M = map(int, input().split())
S = input().strip()

# Split into groups separated by '0's
groups = S.split('0')

max_count = 0
for group in groups:
    cnt = group.count('2')
    if cnt > max_count:
        max_count = cnt

print(max_count)