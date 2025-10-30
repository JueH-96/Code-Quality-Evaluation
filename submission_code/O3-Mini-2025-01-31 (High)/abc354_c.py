def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    # Store cards as tuples: (strength, cost, index)
    cards = []
    for i in range(n):
        a = int(next(it))
        c = int(next(it))
        cards.append((a, c, i + 1))
    
    # Sort the cards by strength (A) in ascending order.
    cards.sort(key=lambda x: x[0])
    
    # We'll iterate from the card with the highest strength downwards.
    survivors = []
    # Initialize with infinity so that the first (highest strength) card always survives.
    min_cost = float('inf')
    
    # Process cards in descending order by strength.
    for a, c, index in reversed(cards):
        # If current card cost is less than the minimal cost found among cards with higher strength,
        # then no card with greater strength has a lower cost, so add it to survivors.
        if c < min_cost:
            survivors.append(index)
            min_cost = c
    
    # The survivors may be computed in arbitrary order. We must output them sorted in ascending order of card indices.
    survivors.sort()
    
    # Prepare and output the result.
    output = []
    output.append(str(len(survivors)))
    output.append(" ".join(map(str, survivors)))
    sys.stdout.write("
".join(output))

if __name__ == '__main__':
    main()