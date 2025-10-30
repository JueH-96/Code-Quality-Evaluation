def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    cards = list(map(int, data))
    
    # Function to check full house condition for a list of 5 cards.
    def is_full_house(five_cards):
        # Count frequencies.
        freq = {}
        for card in five_cards:
            freq[card] = freq.get(card, 0) + 1
        # Must contain exactly 2 distinct numbers
        if len(freq) != 2:
            return False
        # One frequency must be 3 and the other must be 2
        values = sorted(freq.values())
        return values == [2, 3]
    
    # Try adding a candidate card from 1 to 13.
    possible = False
    for candidate in range(1, 14):
        if is_full_house(cards + [candidate]):
            possible = True
            break
    
    print("Yes" if possible else "No")
    
if __name__ == '__main__':
    main()