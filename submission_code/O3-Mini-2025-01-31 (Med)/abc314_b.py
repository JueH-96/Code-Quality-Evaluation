def main():
    import sys
    data = sys.stdin.read().split()
    # Convert all input tokens to integers
    numbers = list(map(int, data))
    if not numbers:
        return
    
    index = 0
    N = numbers[index]
    index += 1
    
    bets = []  # list of tuples (person_number, number_of_bets, set_of_bets)
    
    for person in range(1, N + 1):
        C = numbers[index]
        index += 1
        outcomes = set(numbers[index:index + C])
        index += C
        bets.append((person, C, outcomes))
    
    # The outcome of the spin
    if index < len(numbers):
        X = numbers[index]
    else:
        return
    
    # Find persons who bet on X.
    valid_people = []
    for person, C, outcomes in bets:
        if X in outcomes:
            valid_people.append((person, C))
    
    # No person bet on X.
    if not valid_people:
        print(0)
        print()
        return
    
    # Find the minimum number of bets among persons who bet on X.
    min_bets = min(C for _, C in valid_people)
    
    # Collect persons that have exactly min_bets and sort them by person number.
    result = sorted(person for person, C in valid_people if C == min_bets)
    
    # Output result.
    print(len(result))
    if result:
        print(" ".join(map(str, result)))
    else:
        print()

if __name__ == "__main__":
    main()