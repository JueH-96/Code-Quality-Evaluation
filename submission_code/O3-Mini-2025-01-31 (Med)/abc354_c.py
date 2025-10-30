def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    it = iter(input_data)
    n = int(next(it))
    
    # Store cards as tuples (strength, cost, original_index)
    cards = []
    for i in range(1, n + 1):
        a = int(next(it))
        c = int(next(it))
        cards.append((a, c, i))
        
    # Sort cards in descending order of strength.
    # Note: Since all strengths are distinct, this ordering is strict.
    cards.sort(key=lambda card: -card[0])
    
    survivors = []
    min_cost = 10**10  # Initialize min_cost to a value larger than any possible cost.
    
    # Iterate over cards in descending order of strength.
    # For each card, if its cost is lower than the minimum cost seen so far (from cards with strictly higher strength),
    # then it is not dominated by any card with higher strength.
    for a, c, idx in cards:
        if c < min_cost:
            survivors.append(idx)
            min_cost = c
    
    # Print survivors in sorted order of their original indices.
    survivors.sort()
    
    output_lines = []
    output_lines.append(str(len(survivors)))
    output_lines.append(" ".join(map(str, survivors)))
    sys.stdout.write("
".join(output_lines))

if __name__ == '__main__':
    main()