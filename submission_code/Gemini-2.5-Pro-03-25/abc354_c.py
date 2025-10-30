# YOUR CODE HERE
import sys

def solve():
    # Read the number of cards
    N = int(sys.stdin.readline())
    
    # List to store card data. Each element will be a tuple: (Strength A, Cost C, Original Index)
    cards_data = []
    # Read N cards' data
    for i in range(N):
        # Read Strength A and Cost C for the i-th card (0-indexed loop)
        A, C = map(int, sys.stdin.readline().split())
        # Append the card data along with its original 1-based index (i + 1)
        cards_data.append((A, C, i + 1)) 

    # Sort the cards based on their strength A in ascending order.
    # Since the problem guarantees that all A_i are distinct, sorting only by A is sufficient.
    # The lambda function specifies sorting by the first element of the tuple (which is A).
    cards_data.sort(key=lambda x: x[0])

    # Compute suffix minimum costs C for the sorted list of cards.
    # SufMinC[k] will store the minimum cost among cards from index k to N-1 in the sorted list `cards_data`.
    # The array size is N+1 to simplify boundary conditions. SufMinC[N] will represent the minimum cost
    # for an empty set of cards (conceptually infinity).
    SufMinC = [0] * (N + 1)
    
    # Use a very large integer value to represent infinity.
    # This value must be larger than any possible cost C_i (max 10^9).
    # Using 2 * 10^9 + 7 provides a safe margin.
    infinity_proxy = 2 * 10**9 + 7 
    SufMinC[N] = infinity_proxy 

    # Iterate from right to left (from N-1 down to 0) to compute suffix minimums.
    # SufMinC[k] is the minimum of the cost of the card at index k and the minimum cost
    # of cards from index k+1 onwards (which is SufMinC[k+1]).
    for k in range(N - 1, -1, -1):
        # cards_data[k][1] is the cost C of the k-th card (0-indexed) in the sorted list.
        SufMinC[k] = min(cards_data[k][1], SufMinC[k+1])

    # List to store the original indices of the cards that remain after the discarding process.
    remaining_indices = []
    
    # Iterate through the sorted cards to determine which ones remain.
    for k in range(N):
        # Get the current card's data from the sorted list.
        # The structure is (Strength A, Cost C, Original Index).
        # We only need Cost C and Original Index for the check and storage.
        # _ is used to ignore Strength A as it's implicitly handled by the sorted order and suffix minimum calculation.
        _, current_C, current_idx = cards_data[k]
        
        # M_k represents the minimum cost among cards with strength strictly greater than the current card's strength.
        # Since the list `cards_data` is sorted by strength A, these stronger cards are located at indices k+1 to N-1.
        # The minimum cost among these cards has been precomputed and is stored in SufMinC[k+1].
        min_cost_stronger = SufMinC[k+1]
        
        # A card `y` remains if there is no card `x` such that A_x > A_y and C_x < C_y.
        # This is equivalent to saying that for all cards `x` with A_x > A_y, we must have C_x >= C_y.
        # This condition implies that the minimum cost among all cards stronger than `y` must be greater than or equal to C_y.
        # Let M_y = min{C_x | A_x > A_y}. Card y remains iff M_y >= C_y.
        # For the k-th card in the sorted list, this condition is min_cost_stronger >= current_C.
        # The check `current_C <= min_cost_stronger` implements this condition.
        if current_C <= min_cost_stronger:
             # If the condition holds, the card remains. Add its original index to the list.
             remaining_indices.append(current_idx)

    # The problem requires the output indices to be sorted in ascending order.
    remaining_indices.sort()

    # Print the total count of remaining cards.
    print(len(remaining_indices))
    # Print the sorted indices of the remaining cards, separated by spaces.
    # The * operator unpacks the list `remaining_indices` into individual arguments for the print function.
    print(*(remaining_indices))

# Execute the solve function to run the program based on standard input/output.
solve()