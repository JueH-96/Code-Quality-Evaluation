import sys

def main():
    N = int(sys.stdin.readline())
    
    cards = []
    for i in range(N):
        A, C = map(int, sys.stdin.readline().split())
        # Store as (Strength A, Cost C, Original Index)
        # Original indices are 1-based (i+1).
        cards.append((A, C, i + 1))

    # Sort cards by strength A in ascending order.
    # Python's default tuple sort sorts by the first element, then by the second, etc.
    # Since A_i are distinct, sorting by A is guaranteed by sorting tuples (A, C, id).
    cards.sort() 

    kept_card_indices = []
    
    # This variable will store the minimum cost encountered so far among cards
    # that are stronger than the current card and are themselves kept according to this logic.
    # When processing card_i (iterating from strongest P_{N-1} down to P_0):
    # min_cost_threshold holds the minimum cost among P_{k} for k > i,
    # where P_k are cards that were kept by this algorithm.
    min_cost_threshold = float('inf')

    # Iterate from the card with highest strength (rightmost in sorted list `cards`) downwards.
    # cards[N-1] is the strongest, cards[0] is the weakest.
    for i in range(N - 1, -1, -1):
        # current_A = cards[i][0] # Not used in logic, but part of tuple
        current_C = cards[i][1]
        current_id = cards[i][2]
        
        # A card is kept if its cost (current_C) is less than the minimum cost found so far
        # (min_cost_threshold) among cards that are stronger AND have themselves been kept.
        if current_C < min_cost_threshold:
            kept_card_indices.append(current_id)
            # This card is kept, so its cost becomes the new threshold if it's lower.
            min_cost_threshold = current_C
        # Else (current_C >= min_cost_threshold):
        # (Since costs C_i are distinct, this means current_C > min_cost_threshold)
        # This card is dominated by some stronger card P_k (k>i) whose cost C_k 
        # effectively set the current min_cost_threshold.
        # So, this card is discarded.
        # min_cost_threshold does not change because current_C was not smaller.
            
    # The problem requires indices to be sorted in ascending order.
    # kept_card_indices currently has indices of cards that are kept,
    # but they are not necessarily sorted by index value (they were added based on strength rank).
    kept_card_indices.sort()

    sys.stdout.write(str(len(kept_card_indices)) + "
")
    if len(kept_card_indices) > 0: # Avoid printing an empty line if no cards are kept (though problem constraints imply N>=2)
        sys.stdout.write(" ".join(map(str, kept_card_indices)) + "
")
    else: # Handle N=0 case or if all cards are discarded (might not be possible with N>=2)
        sys.stdout.write("
")


if __name__ == '__main__':
    main()