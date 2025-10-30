n = int(input())
a = list(map(int, input().split()))
groups = []
for i in range(n - 1):
    if a[i] == a[i + 1]:
        groups.append(a[i])

if not groups:
    print(0)
else:
    last_seen = {}
    left = 0
    max_len = 0
    current_max = 0
    for right in range(len(groups)):
        current = groups[right]
        if current in last_seen and last_seen[current] >= left:
            left = last_seen[current] + 1
        last_seen[current] = right
        current_len = right - left + 1
        if current_len > current_max:
            current_max = current_len
    max_length = current_max * 2
    print(max_length)