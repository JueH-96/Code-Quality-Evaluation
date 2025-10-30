L, R = map(int, input().split())

current = L
result = []

while current < R:
    rem = R - current
    if current == 0:
        s_max = 1 << (rem.bit_length() - 1)
    else:
        cnt = 0
        temp = current
        while temp & 1 == 0:
            cnt += 1
            temp >>= 1
        s_max = 1 << cnt
    s = s_max
    while s > rem:
        s >>= 1
    result.append((current, current + s))
    current += s

print(len(result))
for a, b in result:
    print(a, b)