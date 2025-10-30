def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    X = int(input_data[0])
    Y = int(input_data[1])
    Z = int(input_data[2])
    S = input_data[3]
    
    n = len(S)
    INF = 10**18

    # dp[i][0] and dp[i][1] mean: the minimal cost to have printed the first i characters
    # and end with the Caps Lock light off (state 0) or on (state 1).
    # Initially, Caps Lock is off.
    dp_off = 0
    dp_on = INF

    # Process each character one by one. In order to allow toggling
    # whenever it is beneficial (even if not done exactly “during” a letter production),
    # we relax the state before processing each letter.
    for char in S:
        # We can always toggle with cost Z. So for the current step,
        # the minimal cost to be in off state is:
        best_off = dp_off if dp_off < dp_on + Z else dp_on + Z
        best_on  = dp_on if dp_on < dp_off + Z else dp_off + Z

        # Now, in the printing step, depending on our state the keys output:
        # When Caps Lock is off:
        #    • Press 'a' key alone: outputs 'a' at cost X.
        #    • Press 'a' key with Shift: outputs 'A' at cost Y.
        # When Caps Lock is on:
        #    • Press 'a' key alone: outputs 'A' at cost X.
        #    • Press 'a' key with Shift: outputs 'a' at cost Y.

        # Option for remaining in off state:
        if char == 'a':
            # When in off state the normal key gives 'a'
            cost_off = X
        else:
            # Otherwise, we need Shift key to produce 'A'
            cost_off = Y
        candidate_off = best_off + cost_off

        # Option for remaining in on state:
        if char == 'A':
            # When in on state normal key gives 'A'
            cost_on = X
        else:
            # Otherwise, use Shift to get 'a'
            cost_on = Y
        candidate_on = best_on + cost_on

        # Update dp values for the next step (after printing this character)
        dp_off, dp_on = candidate_off, candidate_on

    # The answer is the minimum cost of reaching state off or on.
    result = dp_off if dp_off < dp_on else dp_on
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()