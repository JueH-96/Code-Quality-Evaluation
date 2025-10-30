def main():
    import sys
    data = sys.stdin.read().split()
    # Convert all tokens to integers.
    numbers = list(map(int, data))
    index = 0
    N = numbers[index]
    index += 1
    
    # We'll store bets as list of (person_index, list_of_bets, count)
    bets = []
    for person in range(1, N + 1):
        Ci = numbers[index]
        index += 1
        outcomes = numbers[index:index + Ci]
        index += Ci
        bets.append((person, outcomes, Ci))
    
    X = numbers[index]  # outcome of the spin
    
    # Find persons who bet on outcome X.
    candidates = []
    for person, outcomes, count in bets:
        if X in outcomes:
            candidates.append((person, count))
    
    # If no candidate bet on X, print 0 and empty line.
    if not candidates:
        print(0)
        print()
        return
    
    # Find the minimum number of bets among candidates
    min_count = min(count for _, count in candidates)
    
    # Get persons with count equal to min_count in ascending order.
    selected = sorted(person for person, count in candidates if count == min_count)
    
    # Print the results in the desired format.
    print(len(selected))
    if selected:
        print(" ".join(map(str, selected)))
    else:
        print()

if __name__ == "__main__":
    main()