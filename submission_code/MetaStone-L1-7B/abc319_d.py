n, m = map(int, input().split())
l = list(map(int, input().split()))

def is_possible(W, M, L):
    if not L:
        return True
    lines = 1
    current = L[0]
    for i in range(1, len(L)):
        if current + L[i] + 1 <= W:
            current += L[i] + 1
        else:
            lines += 1
            if lines > M:
                return False
            current = L[i]
    return lines <= M

low = max(l)
high = sum(l) + (n - 1)

while low < high:
    mid = (low + high) // 2
    if is_possible(mid, m, l):
        high = mid
    else:
        low = mid + 1

print(low)