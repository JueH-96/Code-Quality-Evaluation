L, R = map(int, input().split())

result = []
pos = L

while pos < R:
    if pos == 0:
        i = R.bit_length() - 1
    else:
        trailing_zeros = (pos & -pos).bit_length() - 1
        remaining = R - pos
        max_step = remaining.bit_length() - 1
        i = min(trailing_zeros, max_step)
    
    step = 1 << i
    result.append((pos, pos + step))
    pos += step

print(len(result))
for l, r in result:
    print(l, r)