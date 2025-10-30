n, m = map(int, input().split())
L = list(map(int, input().split()))

low = max(L)
high = sum(L) + (n - 1)

def count_lines(W):
    if not L:
        return 0
    lines = 1
    current = L[0]
    for w in L[1:]:
        if current + 1 + w > W:
            lines += 1
            current = w
            if lines > m:
                break
        else:
            current += 1 + w
    return lines

while low < high:
    mid = (low + high) // 2
    if count_lines(mid) <= m:
        high = mid
    else:
        low = mid + 1

print(low)