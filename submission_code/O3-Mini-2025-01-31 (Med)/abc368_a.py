def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    # First two values are N and K
    N = int(input_data[0])
    K = int(input_data[1])
    
    # Next N values are the card numbers from top to bottom
    cards = list(map(int, input_data[2:2+N]))
    
    # Taking K cards from the bottom and putting them on top.
    # The bottom K cards are the last K elements.
    new_order = cards[-K:] + cards[:-K]
    
    # Print the result as space-separated values.
    print(" ".join(map(str, new_order)))
    
if __name__ == "__main__":
    main()