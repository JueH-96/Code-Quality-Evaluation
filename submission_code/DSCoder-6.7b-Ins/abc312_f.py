import sys
from heapq import nlargest

def solve():
    N, M = map(int, sys.stdin.readline().split())
    items = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]

    pull_tab_cans = [x for t, x in items if t == 0]
    regular_cans = [x for t, x in items if t == 1]
    can_openers = [x for t, x in items if t == 2]

    max_pull_tab_can = max(pull_tab_cans) if pull_tab_cans else 0
    max_regular_can = max(regular_cans) if regular_cans else 0
    max_can_opener = max(nlargest(M, can_openers)) if can_openers else 0

    total_happiness = max_pull_tab_can + max_regular_can + max_can_opener

    print(total_happiness)

solve()