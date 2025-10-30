import sys

def main() -> None:
    input = sys.stdin.readline

    # read number of cards
    N = int(input())

    # store (strength, cost, original_index)
    cards = []
    for i in range(1, N + 1):
        a, c = map(int, input().split())
        cards.append((a, c, i))

    # process cards in decreasing order of strength
    cards.sort(key=lambda x: -x[0])          # strongest first

    kept = []                                # indices of the remaining cards
    min_cost_so_far = float('inf')           # lowest cost among stronger cards

    for a, c, idx in cards:
        # if there already exists a stronger card with lower cost, this card is dominated
        if min_cost_so_far < c:
            continue
        # otherwise, keep the card and update the minimum cost seen so far
        kept.append(idx)
        min_cost_so_far = c

    kept.sort()                              # output must be in ascending order
    print(len(kept))
    print(' '.join(map(str, kept)))

if __name__ == "__main__":
    main()