import sys

def solve():
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    
    L = len(S)
    
    # Phase 1: Construct the number by setting '1's from S, 
    # and '0' for '?' and '0's from S.
    # This is the smallest number we can form that respects the '1's in S
    # (effectively, '?' are treated as '0's for this step).
    current_ans = 0
    for i in range(L):
        if S[i] == '1':
            # Calculate the bit position (0-indexed from the right, LSB is 0)
            # S[0] is MSB, S[L-1] is LSB.
            # S[i] corresponds to bit L-1-i.
            bit_pos = L - 1 - i
            current_ans |= (1 << bit_pos)
            
    # If this base number (where '?'s are '0') is already greater than N,
    # then any number formed from S by replacing '?'s will also be > N
    # (since changing '?' from '0' to '1' only increases the value).
    # Thus, no number in T can be <= N.
    if current_ans > N:
        print("-1")
    else:
        # Phase 2: Greedily try to change '?' from '0' to '1' to maximize current_ans,
        # without exceeding N. Iterate from MSB to LSB.
        # current_ans currently has '1's from S fixed, and '0's at positions of '0' in S.
        # The bits corresponding to '?' in S are also '0' in current_ans.
        for i in range(L):
            if S[i] == '?':
                bit_pos = L - 1 - i
                # Try setting this bit (which is currently '0' in current_ans) to '1'.
                # This means ORing with (1 << bit_pos).
                potential_ans = current_ans | (1 << bit_pos)
                
                if potential_ans <= N:
                    # If it's still not too large, make the change.
                    current_ans = potential_ans
                # Else, this '?' must remain '0' to not exceed N. 
                # So, current_ans is not changed.
                    
        print(current_ans)

solve()