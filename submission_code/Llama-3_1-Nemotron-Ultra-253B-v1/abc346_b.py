W, B = map(int, input().split())
base = "wbwbwwbwbwbw"

# Check case 1: substring within a single base
for start in range(len(base)):
    for end in range(start, len(base)):
        substr = base[start:end+1]
        w = substr.count('w')
        b = substr.count('b')
        if w == W and b == B:
            print("Yes")
            exit()

# Precompute prefix and suffix counts
prefix_ws = [0] * len(base)
prefix_bs = [0] * len(base)
for s in range(len(base)):
    substr = base[s:]
    prefix_ws[s] = substr.count('w')
    prefix_bs[s] = substr.count('b')

suffix_ws = [0] * len(base)
suffix_bs = [0] * len(base)
for e in range(len(base)):
    substr = base[:e+1]
    suffix_ws[e] = substr.count('w')
    suffix_bs[e] = substr.count('b')

# Check case 2: spanning multiple bases
for s in range(len(base)):
    for e in range(len(base)):
        total_p_w = prefix_ws[s] + suffix_ws[e]
        total_p_b = prefix_bs[s] + suffix_bs[e]
        req_k_w = W - total_p_w
        req_k_b = B - total_p_b
        if req_k_w < 0 or req_k_b < 0:
            continue
        if req_k_w % 7 != 0 or req_k_b % 5 != 0:
            continue
        k_w = req_k_w // 7
        k_b = req_k_b // 5
        if k_w == k_b and k_w >= 0:
            print("Yes")
            exit()

print("No")