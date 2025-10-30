# YOUR CODE HERE
def rearrange_cards(n, k, cards):
    # Take the last K cards and place them on top
    new_stack = cards[-k:] + cards[:-k]
    return new_stack

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    n = int(data[0])
    k = int(data[1])
    cards = list(map(int, data[2:n+2]))
    
    result = rearrange_cards(n, k, cards)
    print(" ".join(map(str, result)))