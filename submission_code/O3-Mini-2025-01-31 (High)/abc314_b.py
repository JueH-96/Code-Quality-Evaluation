def main():
    import sys
    data = sys.stdin.read().split()
    # Pointer to iterate through tokens in data.
    pos = 0
    
    # Read the number of people.
    N = int(data[pos])
    pos += 1
    
    # Store bets for each person.
    persons = []
    for _ in range(N):
        C = int(data[pos])
        pos += 1
        bets = list(map(int, data[pos: pos + C]))
        pos += C
        persons.append((C, bets))
    
    # Read the outcome X.
    X = int(data[pos])
    
    # Find candidate persons who bet on X.
    candidates = []
    for i, (C, bets) in enumerate(persons):
        if X in bets:
            candidates.append((i + 1, C))
            
    # If no candidate exists, output 0.
    if not candidates:
        print(0)
        print("")
        return
    
    # Determine the minimum count among candidates.
    min_count = min(C for _, C in candidates)
    
    # Gather person numbers that have bet on X and have the fewest bets.
    result = sorted(i for i, C in candidates if C == min_count)
    
    # Output the result
    print(len(result))
    if result:
        print(" ".join(map(str, result)))
    else:
        print("")

# Call the main function.
if __name__ == '__main__':
    main()