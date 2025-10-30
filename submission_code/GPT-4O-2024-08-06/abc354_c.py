# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    cards = []
    
    for i in range(N):
        A = int(data[2 * i + 1])
        C = int(data[2 * i + 2])
        cards.append((A, C, i + 1))
    
    # Sort cards by strength A
    cards.sort()
    
    # To store the indices of remaining cards
    remaining_cards = []
    min_cost = float('inf')
    
    for A, C, index in cards:
        if C < min_cost:
            remaining_cards.append(index)
            min_cost = C
    
    # Sort indices of remaining cards
    remaining_cards.sort()
    
    # Print the result
    print(len(remaining_cards))
    print(' '.join(map(str, remaining_cards)))

if __name__ == "__main__":
    main()