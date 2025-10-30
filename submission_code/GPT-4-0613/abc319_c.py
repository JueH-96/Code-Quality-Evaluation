from collections import Counter
from math import factorial

def solve():
    c = [list(map(int, input().split())) for _ in range(3)]
    cnt = Counter()
    for i in range(3):
        for j in range(3):
            cnt[c[i][j]] += 1
    cnt = sorted(cnt.items(), key=lambda x: -x[1])
    if cnt[0][1] > 4 or cnt[1][1] > 4:
        print(0)
        return
    if cnt[0][1] == 4:
        if cnt[1][1] == 4:
            print(factorial(4)**2 / factorial(8))
        else:
            print(factorial(4) / factorial(8))
        return
    if cnt[0][1] == 3:
        if cnt[1][1] == 3:
            print(factorial(3)**2 * 2 / factorial(8))
        else:
            print(factorial(3) / factorial(8))
        return
    print(1)

solve()