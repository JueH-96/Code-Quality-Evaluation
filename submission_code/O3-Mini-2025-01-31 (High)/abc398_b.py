def main():
    import sys
    # Read the seven integers from standard input.
    data = list(map(int, sys.stdin.read().split()))
    
    # Count the occurrences of each card.
    counts = {}
    for card in data:
        counts[card] = counts.get(card, 0) + 1

    # Check for existence of one card value with at least 3 occurrences (for the three-of-a-kind)
    # and another distinct card value with at least 2 occurrences (for the pair).
    can_form_full_house = False
    for num1, count1 in counts.items():
        if count1 >= 3:
            for num2, count2 in counts.items():
                if num1 != num2 and count2 >= 2:
                    can_form_full_house = True
                    break
        if can_form_full_house:
            break

    print("Yes" if can_form_full_house else "No")

if __name__ == '__main__':
    main()