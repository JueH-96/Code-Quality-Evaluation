n = int(input())
p = list(map(int, input().split()))
p = [x - 1 for x in p]
a = list(map(int, input().split()))

# Preprocess cycles
visited = [False] * n
cycle_list = [[] for _ in range(n)]
index_in_cycle = [0] * n
cycle_length = [0] * n

for i in range(n):
    if not visited[i]:
        cycle = []
        current = i
        while True:
            cycle.append(current)
            visited[current] = True
            current = p[current]
            if current == i:
                break
        for idx, elem in enumerate(cycle):
            cycle_list[elem] = cycle
            index_in_cycle[elem] = idx
            cycle_length[elem] = len(cycle)

# Find candidates from the first element's cycle
cycle_0 = cycle_list[0]
min_val = min(a[pos] for pos in cycle_0)
candidates = [k for k in range(len(cycle_0)) if a[cycle_0[k]] == min_val]

current_candidates = candidates.copy()
result_k = None

for i in range(n):
    if len(current_candidates) == 1:
        result_k = current_candidates[0]
        break
    min_value = None
    new_candidates = []
    for k in current_candidates:
        cycle_i = cycle_list[i]
        idx_i = index_in_cycle[i]
        len_i = cycle_length[i]
        pos_in_cycle = (idx_i + k) % len_i
        val = a[cycle_i[pos_in_cycle]]
        if min_value is None or val < min_value:
            min_value = val
            new_candidates = [k]
        elif val == min_value:
            new_candidates.append(k)
    current_candidates = new_candidates
    if not current_candidates:
        break

# If loop ends without breaking, pick the first candidate
if result_k is None:
    result_k = current_candidates[0] if current_candidates else candidates[0]

# Generate the resulting array
ans = []
for i in range(n):
    cycle_i = cycle_list[i]
    idx_i = index_in_cycle[i]
    len_i = cycle_length[i]
    pos_in_cycle = (idx_i + result_k) % len_i
    elem = cycle_i[pos_in_cycle]
    ans.append(a[elem])

print(' '.join(map(str, ans)))