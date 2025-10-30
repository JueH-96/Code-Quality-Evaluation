# YOUR CODE HERE
import sys

def solve():
    # Read the input string S and integer N
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline())
    
    # Get the length of the string S
    L = len(S)
    
    # Initialize the current value being built. This represents the prefix of the binary number.
    current_val = 0
    
    # Iterate through the characters of S from left to right (Most Significant Bit to Least Significant Bit)
    # The index `i` represents the current bit position (0-based).
    for i in range(L):
        
        # Calculate the potential values if we append 0 or 1 to the current prefix
        val_if_0 = current_val * 2 + 0
        val_if_1 = current_val * 2 + 1
        
        # Check feasibility of choosing '1' for the current bit i.
        # A choice is feasible if it's allowed by S[i] ('1' or '?') AND 
        # the minimum possible value constructed with this choice and subsequent minimal choices 
        # does not exceed N.
        can_choose_1 = False
        # Check if S[i] allows setting the current bit to '1'
        if S[i] == '1' or S[i] == '?':
             # Calculate the minimum possible integer value if we set bit i to 1.
             # Start with the value of the prefix ending in 1.
             min_val_if_1 = val_if_1 
             # To find the minimum possible overall value, append the minimum possible values 
             # for the remaining bits (suffix S[i+1...L-1]).
             # For any subsequent character S[k]: if '1', bit must be 1. If '0' or '?', choose 0 for minimum value.
             for k in range(i + 1, L):
                 if S[k] == '1':
                     min_val_if_1 = min_val_if_1 * 2 + 1
                 else: # S[k] == '0' or S[k] == '?'
                     min_val_if_1 = min_val_if_1 * 2 + 0
             
             # If this minimum possible value starting with prefix ending in '1' is not greater than N, 
             # then choosing '1' for the current bit is potentially valid (feasible).
             if min_val_if_1 <= N:
                 can_choose_1 = True

        # Check feasibility of choosing '0' for the current bit i using similar logic.
        can_choose_0 = False
        # Check if S[i] allows setting the current bit to '0'
        if S[i] == '0' or S[i] == '?':
            # Calculate the minimum possible integer value if we set bit i to 0.
            min_val_if_0 = val_if_0
            # Append minimal values for the suffix.
            for k in range(i + 1, L):
                if S[k] == '1':
                    min_val_if_0 = min_val_if_0 * 2 + 1
                else: # S[k] == '0' or S[k] == '?'
                     min_val_if_0 = min_val_if_0 * 2 + 0
            
            # If this minimum possible value starting with prefix ending in '0' is not greater than N,
            # then choosing '0' for the current bit is potentially valid (feasible).
            if min_val_if_0 <= N:
                can_choose_0 = True

        # Now, decide the value for bit i based on the character S[i] and feasibility checks.
        # We want the GREATEST value <= N, so we prefer '1' over '0' if both are allowed and feasible.
        
        if S[i] == '1':
            # If S[i] specifies '1', we MUST choose 1. Check if this choice is feasible.
            if can_choose_1:
                current_val = val_if_1
            else:
                 # If choosing 1 is required but not feasible (minimum resulting value > N),
                 # then no solution exists that satisfies the constraints.
                 print("-1")
                 return # Exit the function
                 
        elif S[i] == '0':
             # If S[i] specifies '0', we MUST choose 0. Check if this choice is feasible.
             if can_choose_0:
                 current_val = val_if_0
             else:
                 # If choosing 0 is required but not feasible (minimum resulting value > N),
                 # then no solution exists.
                 print("-1")
                 return # Exit the function
             
        else: # S[i] == '?'
            # If S[i] is '?', we have a choice between '0' and '1'.
            # To maximize the final value, we greedily prefer '1' if it's feasible.
            if can_choose_1:
                current_val = val_if_1
            # If '1' is not feasible, we must choose '0'. Check if '0' is feasible.
            elif can_choose_0: 
                current_val = val_if_0
            else: 
                 # If neither '1' nor '0' is feasible for this '?' position, no solution exists.
                 print("-1")
                 return # Exit the function

    # If the loop completes without returning -1, it means we successfully constructed 
    # the largest possible value `current_val` that satisfies the pattern S and is potentially <= N.
    # The checks within the loop ensure that at every step, the choice made leads to a minimum possible
    # final value that does not exceed N. Therefore, the final `current_val` itself must be <= N.
    print(current_val)

# Call the solve function to run the logic
solve()