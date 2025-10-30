def main():
    import sys
    input_line = sys.stdin.read().split()
    if not input_line:
        return
    cards = list(map(int, input_line))
    
    # Count the occurrences of each card
    from collections import Counter
    count = Counter(cards)
    distinct = len(count)
    
    # We are allowed to add one card optimally.
    # A Full House of 5 cards must contain exactly 2 distinct integers,
    # with one integer appearing 3 times and the other appearing 2 times.
    
    # Consider the following cases based on the original 4 cards:
    # Case 1: If there are exactly 2 distinct numbers:
    #   Let the frequencies be a and b (thus a + b = 4).
    #   A valid full house can be formed if:
    #       - They are originally [3,1]: then by adding a card equal to the one with count 1,
    #         we get counts [3,2] -> Full House.
    #       - They are [2,2]: then by adding a card equal to one of them,
    #         we get counts [3,2] -> Full House.
    #   Otherwise, it's not possible.
    #
    # Case 2: If there is 1 distinct number:
    #       {x:4}. Adding another card with x yields 5 of a kind, and a Full House requires exactly two distinct values.
    #
    # Case 3: If there are 3 or more distinct numbers:
    #       In every case, adding one card either yields 3 or more distinct cards,
    #       which cannot be rearranged into exactly a triple and a pair.
    
    if distinct == 2:
        freqs = sorted(count.values())
        if freqs == [1, 3] or freqs == [2, 2]:
            print("Yes")
        else:
            print("No")
    else:
        print("No")

if __name__ == '__main__':
    main()