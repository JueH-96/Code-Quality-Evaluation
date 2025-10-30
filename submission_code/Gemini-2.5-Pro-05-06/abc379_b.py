# YOUR CODE HERE
import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S_str = sys.stdin.readline().strip()

    s_chars = list(S_str)
    strawberries = 0
    
    i = 0
    # The loop must check segments starting at i.
    # A segment starting at i covers indices i, i+1, ..., i+K-1.
    # The last valid index is N-1. So, i+K-1 <= N-1, which means i <= N-K.
    while i <= N - K:
        # Check if s_chars[i ... i+K-1] are all 'O'
        is_segment_healthy = True
        # Iterate from current index i up to i+K-1
        for j in range(i, i + K): 
            if s_chars[j] == 'X':
                is_segment_healthy = False
                break
        
        if is_segment_healthy:
            strawberries += 1
            # These K teeth develop cavities
            for j in range(i, i + K): # Iterate from current index i up to i+K-1
                s_chars[j] = 'X'
            
            # Advance by K positions. The teeth s_chars[i...i+K-1] are now 'X'.
            # Any segment starting at an index between i+1 and i+K-1 would
            # overlap with the just-used segment, so it cannot be all 'O's.
            # The earliest a new K-healthy segment could start is i+K.
            i += K
        else:
            # The segment s_chars[i...i+K-1] is not K-healthy.
            # Try starting the segment one position to the right.
            i += 1
            
    print(strawberries)

# Call the solve function to run the solution
solve()