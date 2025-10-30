# Read input
T = input().strip()
N = int(input())
bags = []
for _ in range(N):
    parts = input().split()
    A_i = int(parts[0])
    strings = parts[1:]
    bags.append(strings)

len_T = len(T)
INF = float('inf')

# Initialize DP
previous = [INF] * (len_T + 1)
previous[0] = 0

for bag in bags:
    current = previous.copy()
    for s in bag:
        l = len(s)
        for k in range(len_T + 1):
            if previous[k] == INF:
                continue
            new_j = k + l
            if new_j > len_T:
                continue
            if T[k:new_j] == s:
                if current[new_j] > previous[k] + 1:
                    current[new_j] = previous[k] + 1
    previous = current

result = previous[len_T]
print(result if result != INF else -1)