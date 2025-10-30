import math

K = int(input())

def find_kth(K):
    d = 1
    total = 0
    while True:
        cnt = 0
        for m in range(1, 10):
            cnt += math.comb(m, d-1)
        if total + cnt >= K:
            K -= total
            break
        total += cnt
        d += 1

    res = []
    max_digit = 9
    for _ in range(d):
        for m in range(max_digit, 0, -1):
            remaining = d - len(res) - 1
            c = math.comb(m, remaining)
            if K > c:
                K -= c
            else:
                res.append(str(m))
                max_digit = m - 1
                break
    return ''.join(res)

print(find_kth(K))