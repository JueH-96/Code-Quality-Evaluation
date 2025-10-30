L, R = map(int, input().split())
current = L
result = []
while current < R:
    s_candidate = R - current
    if s_candidate == 0:
        break
    s_start = 1 << (s_candidate.bit_length() - 1)
    s = None
    while s_start >= 1:
        if current % s_start == 0:
            s = s_start
            break
        s_start //= 2
    result.append((current, current + s))
    current += s
print(len(result))
for pair in result:
    print(pair[0], pair[1])