# Read input
L, R = map(int, input().split())

current = L
result = []

while current < R:
    if current == 0:
        y = R - current
        step_r = 1 << (y.bit_length() - 1)
        max_step = step_r
    else:
        step_x = current & -current
        y = R - current
        step_r = 1 << (y.bit_length() - 1)
        max_step = min(step_x, step_r)
    next_end = current + max_step
    result.append((current, next_end))
    current = next_end

# Output the result
print(len(result))
for l, r in result:
    print(l, r)