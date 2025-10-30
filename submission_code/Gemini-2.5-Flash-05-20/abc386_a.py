import collections

def solve():
    # Read the four integers A, B, C, D from a single line of input.
    # map(int, input().split()) converts the space-separated string into integers.
    A, B, C, D = map(int, input().split())
    
    # Put the four cards into a list.
    cards = [A, B, C, D]
    
    # Use collections.Counter to efficiently count the frequency of each card.
    # For example, if cards is [7, 7, 7, 1], freq_counter will be Counter({7: 3, 1: 1}).
    freq_counter = collections.Counter(cards)
    
    # Get the list of frequencies (counts) from the Counter object's values.
    # For example, if freq_counter is Counter({7: 3, 1: 1}), counts might be [3, 1] or [1, 3] depending on internal hash order.
    counts = list(freq_counter.values())
    
    # Sort the frequencies. This makes the comparison consistent regardless of the order
    # in which numbers appeared or were stored in the Counter.
    # For example, [3, 1] becomes [1, 3], and [2, 2] remains [2, 2].
    counts.sort()
    
    # A Full House consists of five cards: three cards of one rank (e.g., three 7s)
    # and two cards of another distinct rank (e.g., two 1s).
    # After adding one card to the initial four, the final set of five cards
    # must have frequencies that, when sorted, look like [2, 3].
    
    # We analyze the sorted frequencies of the initial four cards:
    
    # Case 1: Initial cards are like X, X, X, Y (e.g., 7, 7, 7, 1)
    # The sorted frequencies will be [1, 3].
    # By adding another 'Y' (the card with count 1), we form a Full House (3 of X, 2 of Y).
    if counts == [1, 3]:
        print("Yes")
    
    # Case 2: Initial cards are like X, X, Y, Y (e.g., 3, 3, 5, 5)
    # The sorted frequencies will be [2, 2].
    # By adding another 'X' or another 'Y', we can form a Full House (e.g., 3 of X, 2 of Y).
    elif counts == [2, 2]:
        print("Yes")
        
    # All other possible frequency patterns for 4 cards cannot form a Full House
    # by adding just one card to reach the [2, 3] pattern:
    # - [1, 1, 1, 1] (four distinct cards)
    # - [1, 1, 2] (one pair, two distinct singles)
    # - [4] (four of a kind)
    else:
        print("No")

# Call the solve function to execute the program.
solve()