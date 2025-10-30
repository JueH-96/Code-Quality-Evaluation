from collections import defaultdict

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    left = defaultdict(int)
    right = defaultdict(int)
    for a in A:
        right[a] += 1
    ans = 0
    for a in A:
        right[a] -= 1
        if right[a] > 0:
            ans += right[a]
        left[a] += 1
        if left[a] > 1:
            ans -= left[a] - 1
    print(ans)

solve()