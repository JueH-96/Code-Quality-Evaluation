from collections import Counter

def solve():
    t = int(input())
    for _ in range(t):
        n = int(input())
        a = list(map(int, input().split()))
        counter = Counter(a)
        ans = 0
        for count in counter.values():
            ans += count * (count - 1) // 2
        print(ans)

solve()