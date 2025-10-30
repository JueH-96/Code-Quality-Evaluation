# YOUR CODE HERE

from collections import defaultdict

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    count = defaultdict(int)
    for i in a:
        count[i] += 1

    total = 0
    for v in count.values():
        total += v * (v - 1) * (v - 2) // 6

    print(total)

solve()