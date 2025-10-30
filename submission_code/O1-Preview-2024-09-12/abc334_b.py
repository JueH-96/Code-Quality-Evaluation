A, M, L, R = map(int, input().split())

def ceil_div(a, b):
    q, r = divmod(a, b)
    if r != 0 and ((b > 0 and r > 0) or (b < 0 and r < 0)):
        return q + 1
    return q

k_low = ceil_div(L - A, M)
k_high = (R - A) // M

if k_high < k_low:
    print(0)
else:
    print(k_high - k_low + 1)