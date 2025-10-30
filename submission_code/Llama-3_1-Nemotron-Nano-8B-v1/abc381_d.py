n, *rest = map(int, open(0).read().split())
A = rest[:n] if rest else []

pairs = []
for i in range(n - 1):
    if A[i] == A[i + 1]:
        pairs.append((i, A[i]))

groups = []
current_group = []
for pos, val in pairs:
    if not current_group:
        current_group.append((pos, val))
    else:
        last_pos = current_group[-1][0]
        if pos == last_pos + 2:
            current_group.append((pos, val))
        else:
            groups.append(current_group)
            current_group = [(pos, val)]
if current_group:
    groups.append(current_group)

max_len = 0
for group in groups:
    vals = [v for p, v in group]
    left = 0
    count = {}
    current_max = 0
    for right in range(len(vals)):
        val = vals[right]
        while val in count and count[val] > 0:
            left_val = vals[left]
            count[left_val] -= 1
            if count[left_val] == 0:
                del count[left_val]
            left += 1
        count[val] = count.get(val, 0) + 1
        current_window_size = right - left + 1
        if current_window_size > current_max:
            current_max = current_window_size
    candidate = current_max * 2
    if candidate > max_len:
        max_len = candidate

print(max_len)