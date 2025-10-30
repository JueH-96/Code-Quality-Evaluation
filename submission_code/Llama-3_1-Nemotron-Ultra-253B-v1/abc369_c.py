n, *rest = map(int, open(0).read().split())
if n == 0:
    print(0)
else:
    a = rest[:n]
    total = 1
    prev_diff = None
    current_length = 1
    for i in range(1, n):
        d = a[i] - a[i-1]
        if d == prev_diff:
            current_length += 1
        else:
            current_length = 2
            prev_diff = d
        total += current_length
    print(total)