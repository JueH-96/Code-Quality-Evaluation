import sys

def solve():
    N = int(sys.stdin.readline().strip())
    cards = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N)]
    cards.sort(key=lambda x: (x[1], -x[0]))
    res = [cards[0][0]]
    max_strength = cards[0][0]
    for i in range(1, N):
        if cards[i][0] > max_strength and cards[i][1] < cards[i-1][1]:
            res.append(cards[i][0])
            max_strength = cards[i][0]
    print(len(res))
    print(' '.join(map(str, res)))

solve()