n, *rest = map(int, open(0).read().split())
if n == 0:
    print(0)
else:
    a = rest[:n]
    total = 1
    prev_diff = None
    prev_length = 1
    for i in range(1, n):
        current_diff = a[i] - a[i-1]
        if current_diff == prev_diff:
            current_length = prev_length + 1
        else:
            current_length = 2
        total += current_length
        prev_diff = current_diff
        prev_length = current_length
    print(total)