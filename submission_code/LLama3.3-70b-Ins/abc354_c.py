import sys

def solve():
    n = int(input())
    cards = []
    for i in range(n):
        a, c = map(int, input().split())
        cards.append((a, c, i + 1))

    cards.sort(key=lambda x: (x[0], -x[1]))

    ans = []
    for i in range(n):
        is_discarded = False
        for j in range(i):
            if cards[j][0] > cards[i][0] and cards[j][1] < cards[i][1]:
                is_discarded = True
                break
        if not is_discarded:
            ans.append(cards[i][2])

    print(len(ans))
    print(*ans)

if __name__ == "__main__":
    solve()