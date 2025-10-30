def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # We'll use a dynamic programming approach with 2 states:
    # dp0: maximum total experience with an even number of defeated monsters so far.
    # dp1: maximum total experience with an odd number of defeated monsters so far.
    # Initially, no monster is defeated so dp0 = 0 and dp1 should be very low.
    
    dp0 = 0
    dp1 = -10**18  # a very small negative number to represent an impossible state
    
    for x in A:
        # We decide for the current monster either to skip it or defeat it.
        # When skipping, the dp state remains unchanged.
        # If we defeat it:
        #   - If current state is dp0 (even count so far), the current monster becomes the first (or any odd-index),
        #     and we only gain x (no bonus). The new state becomes odd.
        #   - If current state is dp1 (odd count so far), defeating the monster gives an extra bonus x (total of 2*x),
        #     and the new state becomes even.
        
        nxt0 = dp0  # We'll keep transitions for dp0
        nxt1 = dp1  # We'll keep transitions for dp1
        
        # From state dp0 (even defeated so far) if we defeat this monster:
        nxt1 = max(nxt1, dp0 + x)
        # From state dp1 (odd defeated so far) if we defeat this monster:
        nxt0 = max(nxt0, dp1 + 2 * x)
        
        # Also propagate skipping (which is implicitly done by carrying dp0 and dp1)
        dp0, dp1 = nxt0, nxt1
    
    # The answer is the maximum experience we can achieve regardless of parity.
    answer = max(dp0, dp1)
    sys.stdout.write(str(answer))
    
if __name__ == '__main__':
    main()