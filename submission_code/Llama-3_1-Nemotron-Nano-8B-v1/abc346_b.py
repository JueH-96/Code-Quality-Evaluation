base = "wbwbwwbwbwbw"

# Precompute counts for all possible (s, e)
counts = [[(0, 0) for _ in range(14)] for __ in range(14)]

for s in range(14):
    for e in range(14):
        if s <= e:
            substr = base[s:e+1]
        else:
            substr = base[s:] + base[:e+1]
        w = substr.count('w')
        b = substr.count('b')
        counts[s][e] = (w, b)

# Check all possible (s, e) pairs
W, B = map(int, input().split())
found = False
for s in range(14):
    for e in range(14):
        w_p, b_p = counts[s][e]
        if W < w_p or B < b_p:
            continue
        delta_w = W - w_p
        delta_b = B - b_p
        if delta_w % 8 != 0 or delta_b % 6 != 0:
            continue
        k_w = delta_w // 8
        k_b = delta_b // 6
        if k_w == k_b and k_w >= 0:
            found = True
            break
    if found:
        break

print("Yes" if found else "No")