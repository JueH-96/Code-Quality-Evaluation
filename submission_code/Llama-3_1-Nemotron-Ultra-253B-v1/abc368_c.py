n, *rest = map(int, open(0).read().split())
H = rest[:n]

current_T = 0
for h in H:
    s = current_T
    m = (-s) % 3
    low = 0
    high = h
    ans = high
    while low <= high:
        mid = (low + high) // 2
        if m == 0:
            cnt = mid // 3
        else:
            cnt = (mid - m + 3) // 3
        total = mid + 2 * cnt
        if total >= h:
            ans = mid
            high = mid - 1
        else:
            low = mid + 1
    current_T += ans
print(current_T)