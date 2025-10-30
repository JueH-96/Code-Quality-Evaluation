# YOUR CODE HERE
from collections import defaultdict

def solve():
    N = int(input())
    cards = [list(map(int, input().split())) for _ in range(N)]
    A = [a for a, b in cards]
    B = [b for a, b in cards]
    cnt = defaultdict(int)
    for a, b in cards:
        cnt[(a, b)] += 1
    odd = sum(v % 2 for v in cnt.values())
    even = N - odd
    if odd <= 1:
        print("Takahashi")
    else:
        print("Aoki")

solve()