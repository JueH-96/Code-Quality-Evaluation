# YOUR CODE HERE

import sys

def solve():
    n = int(input())
    cards = []
    for _ in range(n):
        a, c = map(int, input().split())
        cards.append((a, c))

    cards.sort(key=lambda x: (x[0], -x[1]))

    result = [cards[0]]
    for card in cards[1:]:
        if card[1] > result[-1][1]:
            result.append(card)

    print(len(result))
    print(' '.join(str(card[0]) for card in result))

solve()