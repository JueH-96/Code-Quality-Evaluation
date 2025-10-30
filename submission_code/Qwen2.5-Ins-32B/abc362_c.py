import sys

def solve():
    N = int(input())
    LR = [list(map(int, input().split())) for _ in range(N)]
    pos, neg = [], []
    total = 0
    for l, r in LR:
        if l <= 0 <= r:
            total += 0
        elif l > 0:
            pos.append((l, r))
            total += l
        else:
            neg.append((l, r))
            total += r
    if total != 0:
        if total > 0:
            pos.sort(key=lambda x: x[0])
            for l, r in pos:
                if l > 0:
                    if total - l <= 0:
                        total = 0
                        break
                    else:
                        total -= l
                        l += 1
                if total == 0:
                    break
        else:
            neg.sort(key=lambda x: x[1], reverse=True)
            for l, r in neg:
                if r < 0:
                    if total - r >= 0:
                        total = 0
                        break
                    else:
                        total -= r
                        r -= 1
                if total == 0:
                    break
    if total != 0:
        print("No")
    else:
        print("Yes")
        ans = [0] * N
        idx = 0
        for l, r in LR:
            if l <= 0 <= r:
                ans[idx] = 0
            elif l > 0:
                ans[idx] = l
            else:
                ans[idx] = r
            idx += 1
        print(*ans)

solve()