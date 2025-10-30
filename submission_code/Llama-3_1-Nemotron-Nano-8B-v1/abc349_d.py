L, R = map(int, input().split())

current = L
ans = []

while current < R:
    if current == 0:
        s = R - current
        if s == 0:
            break
        s_candidate = 1 << (s.bit_length() - 1)
    else:
        # Compute trailing zeros of current
        k = 0
        temp = current
        while (temp & 1) == 0:
            k += 1
            temp >>= 1
        s = R - current
        if s == 0:
            break
        max_m = min(k, (s).bit_length() - 1)
        s_candidate = 1 << max_m
    ans.append((current, current + s_candidate))
    current += s_candidate

print(len(ans))
for pair in ans:
    print(pair[0], pair[1])