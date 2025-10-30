import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    
    if K == 0:
        print(0)
        return

    # A_list contains the colors of the K socks that are single.
    # These are A_1, ..., A_K from the problem statement, already sorted.
    A_list_values = list(map(int, sys.stdin.readline().split()))

    # dp_even[i]: min weirdness for first i socks from A_list_values (A_list_values[0]...A_list_values[i-1]),
    # assuming all i socks are paired. Meaningful if i is even.
    # dp_odd[i]: min weirdness for first i socks, assuming i-1 socks paired, one left out.
    # Meaningful if i is odd.
    
    # dp arrays are 0-indexed by number of items in prefix. Max K items.
    dp_even = [float('inf')] * (K + 1)
    dp_odd = [float('inf')] * (K + 1)

    # Base case: For 0 socks (empty prefix)
    dp_even[0] = 0 # Cost to pair all 0 socks is 0.
                   # dp_odd[0] remains inf (cannot leave one out from zero socks).

    for i in range(1, K + 1):
        # current_A_val is the i-th sock in 1-based indexing (A_i in math notation).
        # In 0-indexed A_list_values, this is A_list_values[i-1].
        current_A_val = A_list_values[i-1] 
        
        if i == 1:
            # For the first sock (A_list_values[0]):
            # It's impossible to pair it if it's the only one (dp_even[1] remains inf).
            # It must be left out. Cost is 0.
            dp_odd[1] = 0 
        else: # i >= 2
            # The sock before current_A_val is A_list_values[i-2].
            prev_A_val = A_list_values[i-2]
            # Weirdness if A_list_values[i-1] and A_list_values[i-2] are paired
            diff = current_A_val - prev_A_val 

            if i % 2 == 0: # Current prefix length i is even. We calculate dp_even[i].
                # To pair all i socks:
                # A_list_values[i-1] (current) must be paired with A_list_values[i-2] (previous).
                # The first i-2 socks (A_list_values[0]...A_list_values[i-3]) must also all be paired.
                # Cost is dp_even[i-2] + diff.
                if dp_even[i-2] != float('inf'): # Check if state dp_even[i-2] is reachable
                    dp_even[i] = dp_even[i-2] + diff
                
                # dp_odd[i] remains inf because i is even.
            else: # Current prefix length i is odd. We calculate dp_odd[i].
                # To leave one sock out from the first i socks:
                
                # Option 1: The current sock A_list_values[i-1] is left out.
                #   The first i-1 socks (A_list_values[0]...A_list_values[i-2]) must all be paired.
                #   Cost is dp_even[i-1]. (Note: i-1 is even, so dp_even[i-1] is a valid state).
                cost1 = float('inf')
                if dp_even[i-1] != float('inf'): # Check if state dp_even[i-1] is reachable
                    cost1 = dp_even[i-1]
                
                # Option 2: The current sock A_list_values[i-1] is paired with A_list_values[i-2].
                #   Then one sock from the first i-2 socks (A_list_values[0]...A_list_values[i-3]) must be left out.
                #   Cost is dp_odd[i-2] + diff. (Note: i-2 is odd, so dp_odd[i-2] is a valid state).
                cost2 = float('inf')
                if dp_odd[i-2] != float('inf'): # Check if state dp_odd[i-2] is reachable
                    cost2 = dp_odd[i-2] + diff
                
                dp_odd[i] = min(cost1, cost2)

                # dp_even[i] remains inf because i is odd.
    
    # The final answer depends on whether K (total number of single socks) is even or odd.
    # If K is even, all K socks are paired.
    # If K is odd, K-1 socks are paired and one is left out.
    if K % 2 == 0:
        print(dp_even[K])
    else:
        print(dp_odd[K])

solve()