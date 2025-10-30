def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    cards = []
    
    # Read the cards (A_i, C_i), keep track of original index
    idx = 1
    for i in range(N):
        A_i = int(input_data[idx]); C_i = int(input_data[idx+1])
        idx += 2
        cards.append((A_i, C_i, i+1))
        
    # Sort cards by ascending strength A
    cards.sort(key=lambda x: x[0])
    
    # We'll scan from right to left to find which cards survive.
    # Keep track of the smallest cost found so far.
    min_cost = float('inf')
    remain = []
    
    # Traverse from strongest to weakest
    for i in range(N-1, -1, -1):
        A_i, C_i, original_index = cards[i]
        if C_i < min_cost:
            remain.append(original_index)
            min_cost = C_i
        # if not, this card is dominated and discarded
    
    # We collected surviving cards in descending order of A,
    # so sort them by their original indices in ascending order
    remain.sort()
    
    # Print results
    print(len(remain))
    print(" ".join(map(str, remain)))

# Do not forget to call main()
main()