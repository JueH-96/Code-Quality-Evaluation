def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    cards = list(map(int, input_data[2:]))
    
    # Take K cards from the bottom and place them on top
    new_order = cards[-K:] + cards[:-K]
    
    print(" ".join(map(str, new_order)))

if __name__ == "__main__":
    main()