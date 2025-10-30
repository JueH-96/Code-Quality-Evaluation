# YOUR CODE HERE
import sys
from collections import defaultdict

def solve():
    N, M = map(int, input().split())
    items = [tuple(map(int, input().split())) for _ in range(N)]

    pull_tabs = []
    regular_cans = []
    can_openers = []

    for t, x in items:
        if t == 0:
            pull_tabs.append(x)
        elif t == 1:
            regular_cans.append(x)
        else:
            can_openers.append(x)

    pull_tabs.sort(reverse=True)
    regular_cans.sort(reverse=True)
    can_openers.sort(reverse=True)

    happiness = sum(pull_tabs[:M])
    remaining = M - len(pull_tabs)

    if remaining <= 0:
        print(happiness)
        return

    opener_capacity = sum(can_openers)
    usable_cans = min(remaining, opener_capacity, len(regular_cans))

    happiness += sum(regular_cans[:usable_cans])

    print(happiness)

solve()