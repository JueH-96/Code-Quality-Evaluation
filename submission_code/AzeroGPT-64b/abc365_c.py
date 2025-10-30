def main():
    from bisect import bisect_right
    from typing import List
    import sys
    input = sys.stdin.readline

    def hasOverBudget(arr, budget, x):
        surplus, cnt = 0, 0
        for a in arr:
            if x < a:
                surplus += a - x
            else:
                cnt += 1
        if cnt * x >= budget:
            return True
        if surplus <= budget - cnt * x:
            return False
        return True

    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    if sum([min(a, M) for a in A]) <= M:
        print("infinite")
        return

    high = sum(A)
    low = 0

    while high - low > 1:
        med = (high + low) // 2
        if hasOverBudget(A, M, med):
            high = med
        else:
            low = med

    print(low)

main()