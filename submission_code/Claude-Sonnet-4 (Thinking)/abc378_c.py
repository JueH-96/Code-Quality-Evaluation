n = int(input())
a = list(map(int, input().split()))

last_seen = {}
b = []

for i in range(n):
    value = a[i]
    if value in last_seen:
        b.append(last_seen[value] + 1)  # +1 for 1-indexed position
    else:
        b.append(-1)
    
    last_seen[value] = i  # Update last seen position (0-indexed)

print(' '.join(map(str, b)))