def main():
    from bisect import bisect
    from collections import deque
    import sys

    input = lambda: sys.stdin.readline().rstrip()

    N, K = map(int, input().split())
    A = list(map(int, input().split()))

    vals = []
    for a in A:
        vals.append(a)
        if len(vals) > K:
            vals.pop(0)
        if len(vals) == K:
            print(min(vals) - max(vals))
            return
    vals.sort()

    mque = deque([vals[K]])
    for a in reversed(vals[K:]):
        mque.appendleft(a)

    ans = vals[-1] - vals[0]
    for i in range(K, 2 * K):
        idx = bisect(vals, mque[K - 1])
        if idx >= 2 * K + 1:
            continue
        if idx >= i + 2:
            cval = max(vals[:i+1]) - mque[K - 1]
            if cval < ans:
                ans = cval
        else:
            mque.appendleft(vals[idx])
            cval = mque[-1] - mque[0]
            if cval < ans:
                ans = cval

        if len(mque) == K + 1 and mque[-1] == vals[i - K]:
            mque.pop()

        cval = mque[-1] - vals[i-K+1]
        if cval < ans:
            ans = cval

    print(ans)

main()