import math

def solve():
    n = int(input())
    q = list(map(int, input().split()))
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    def is_feasible(s):
        min_a = 0
        max_a = s
        for i in range(n):
            if a[i] > b[i]:
                if a[i] - b[i] > 0:
                    max_val = (q[i] - s * b[i]) // (a[i] - b[i])
                    max_a = min(max_a, max_val)
            elif a[i] < b[i]:
                if b[i] - a[i] > 0:
                    num = q[i] - s * b[i]
                    den = a[i] - b[i]
                    if den != 0:
                        min_val = math.ceil(num / den)
                        min_a = max(min_a, min_val)
            elif a[i] == b[i]:
                if a[i] > 0:
                    if s * a[i] > q[i]:
                        return False
        return min_a <= max_a

    low = 0
    high = sum(q) + 1
    ans = 0
    while low <= high:
        mid = (low + high) // 2
        if is_feasible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    print(ans)