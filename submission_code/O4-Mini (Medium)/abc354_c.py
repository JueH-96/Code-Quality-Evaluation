def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    cards = []
    for i in range(1, N+1):
        A, C = map(int, input().split())
        cards.append((A, C, i))

    # Sort cards by strength A in ascending order
    cards.sort(key=lambda x: x[0])

    # Extract costs in the sorted order
    costs = [cards[i][1] for i in range(N)]

    # Build suffix minimum array of costs:
    # suffix_min[i] = min(costs[i], costs[i+1], ..., costs[N-1])
    INF = 10**18
    suffix_min = [INF] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_min[i] = min(costs[i], suffix_min[i + 1])

    # A card at position i is dominated if there exists a card with higher A
    # (i.e. in positions > i) that has a strictly lower cost.
    remaining = []
    for i in range(N):
        # if the minimum cost among cards with larger A is < costs[i],
        # then cards[i] is dominated and discarded
        if suffix_min[i + 1] < costs[i]:
            continue
        remaining.append(cards[i][2])

    # Output the result: number of remaining cards and their original indices
    remaining.sort()
    print(len(remaining))
    print(*remaining)


if __name__ == "__main__":
    main()