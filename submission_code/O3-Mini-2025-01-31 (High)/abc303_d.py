def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return

    # Parse the inputs
    X = int(data[0])
    Y = int(data[1])
    Z = int(data[2])
    S = data[3].strip()
    
    # We use DP with two states: 
    # state 0 means Caps Lock is OFF, and state 1 means Caps Lock is ON.
    # Initially, the Caps Lock (state 0) is OFF.
    # dp0: minimum cost to have processed a prefix with current state = 0.
    # dp1: minimum cost to have processed a prefix with current state = 1.
    # We initialize:
    INF = 10**18
    dp0, dp1 = 0, INF  # initially, state is OFF so dp0 = 0; state ON is not possible initially.
    
    # For each character, we have a cost function f(s, letter):
    # When state==0 (Caps off), pressing only 'a' gives 'a' at cost X, shift+'a' gives 'A' at cost Y.
    # When state==1 (Caps on), pressing only 'a' gives 'A' at cost X, shift+'a' gives 'a' at cost Y.
    # We can also toggle the caps lock (cost Z) before pressing.
    #
    # So for each letter, if we want the final state to be s (0 or 1), we have:
    #   new_dp[s] = min( dp[s] + f(s, letter),   dp[1-s] + Z + f(s, letter) )
    
    for ch in S:
        # Calculate the cost to produce the given character ch when in state 0 or state 1.
        # In state 0, natural output is 'a', so:
        #    if we need 'a', cost = X, and if we need 'A', cost = Y.
        # In state 1, natural output is 'A', so:
        #    if we need 'A', cost = X, and if we need 'a', cost = Y.
        if ch == 'a':
            cost0 = X  # in state 0, plain key gives 'a'
            cost1 = Y  # in state 1, must use shift to get 'a'
        else:  # ch == 'A'
            cost0 = Y  # in state 0, use shift key to get 'A'
            cost1 = X  # in state 1, plain key gives 'A'
        
        new_dp0 = min(dp0 + cost0, dp1 + Z + cost0)
        new_dp1 = min(dp1 + cost1, dp0 + Z + cost1)
        dp0, dp1 = new_dp0, new_dp1

    # The answer is the minimum cost regardless of the final state.
    ans = min(dp0, dp1)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()