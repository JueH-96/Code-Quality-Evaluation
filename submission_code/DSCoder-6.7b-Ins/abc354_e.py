import sys
from collections import defaultdict

def solve():
    N = int(sys.stdin.readline().strip())
    cards = []
    for _ in range(N):
        a, b = map(int, sys.stdin.readline().strip().split())
        cards.append((a, b))

    front_dict = defaultdict(list)
    back_dict = defaultdict(list)
    for i, (a, b) in enumerate(cards):
        front_dict[a].append(i)
        back_dict[b].append(i)

    front_matches = [False] * N
    back_matches = [False] * N
    for a, indices in front_dict.items():
        if len(indices) >= 2:
            for i in indices:
                front_matches[i] = True

    for b, indices in back_dict.items():
        if len(indices) >= 2:
            for i in indices:
                back_matches[i] = True

    front_possible = any(front_matches[i] for i in range(N) if not back_matches[i])
    back_possible = any(back_matches[i] for i in range(N) if not front_matches[i])

    if front_possible and back_possible:
        print("Takahashi")
    else:
        print("Aoki")

solve()