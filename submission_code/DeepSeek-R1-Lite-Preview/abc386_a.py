from collections import Counter
import sys

def main():
    # Read input
    input_line = sys.stdin.read().strip()
    if not input_line:
        print("No")
        return
    parts = input_line.split()
    if len(parts) != 4:
        print("No")
        return
    try:
        A, B, C, D = map(int, parts)
    except:
        print("No")
        return
    # List of the four cards
    cards = [A, B, C, D]
    # Possible card values to add
    possible_additions = range(1, 14)
    # Check each possible addition
    for add in possible_additions:
        new_cards = cards + [add]
        freq = Counter(new_cards)
        # Check if there is exactly one value with count 3 and another with count 2
        counts = freq.values()
        if 3 in counts and 2 in counts:
            print("Yes")
            return
    # If no addition leads to a Full House
    print("No")

if __name__ == "__main__":
    main()