def main():
    import sys
    input = sys.stdin.readline

    n = int(input())
    cards = []
    for i in range(n):
        a, c = map(int, input().split())
        # store: original index (1-indexed), strength, cost
        cards.append((i + 1, a, c))
    
    # We want to eliminate card y if there exists card x such that:
    # A_x > A_y and C_x < C_y.
    # This is equivalent to "card y is dominated" by a card with higher strength and lower cost.
    # Notably, the final surviving cards are those which are not dominated.
    # Sort by strength descending so that any card processed earlier has strictly larger strength.
    # Then, for current card, if its cost is greater than the best (lowest) cost seen so far among cards with larger strength,
    # it is dominated and discarded.
    cards.sort(key=lambda x: x[1], reverse=True)  # sort by strength descending
    survivors = []
    min_cost = float('inf')
    for idx, a, c in cards:
        # For card to be safe, we must not have any card with higher strength and lower cost.
        if c <= min_cost:
            survivors.append((idx, a, c))
            min_cost = c  # update the minimum cost seen so far
    # survivors may not be in original order, we need to output in increasing order of card index.
    survivors.sort(key=lambda x: x[0])
    # Output the number of survivors and their indices.
    print(len(survivors))
    print(" ".join(str(x[0]) for x in survivors))

if __name__ == '__main__':
    main()