# YOUR CODE HERE
def rearrange_cards():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:]))
    
    # Take K cards from the bottom and place them on top
    bottom_cards = A[-K:]
    remaining_cards = A[:-K]
    
    # New order of cards
    new_order = bottom_cards + remaining_cards
    
    # Print the result
    print(' '.join(map(str, new_order)))

rearrange_cards()