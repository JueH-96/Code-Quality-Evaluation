INF = float('inf')

T = input().strip()
N = int(input())
bags = []
for _ in range(N):
    parts = input().split()
    A_i = int(parts[0])
    strings = parts[1:1+A_i]
    bags.append(strings)

len_T = len(T)
previous = [INF] * (len_T + 1)
previous[0] = 0  # Base case: 0 cost to be at position 0 before any bags

for bag in bags:
    current = list(previous)  # Initialize with do-nothing costs
    for s in bag:
        length = len(s)
        max_k = len_T - length
        if max_k < 0:
            continue  # String longer than T, skip
        for k in range(max_k + 1):
            if T[k:k+length] == s:
                j = k + length
                if previous[k] + 1 < current[j]:
                    current[j] = previous[k] + 1
    previous = current

print(previous[len_T] if previous[len_T] != INF else -1)