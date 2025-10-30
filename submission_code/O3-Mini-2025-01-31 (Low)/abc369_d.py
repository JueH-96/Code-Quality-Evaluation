def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    n = int(input_data[0])
    A = list(map(int, input_data[1:]))
    
    # dp0: maximum xp obtainable when the number of defeated monsters so far is even.
    # dp1: maximum xp obtainable when the number of defeated monsters so far is odd.
    # Initially, no monsters have been defeated so dp0 = 0 and dp1 is impossible.
    dp0 = 0
    dp1 = -10**18  # A very small number to represent negative infinity.
    
    # Process monsters in order.
    for x in A:
        # If we skip the monster, the dp values remain unchanged.
        # If we defeat the monster:
        #   - From dp0 (even), defeat becomes the 1st, 3rd, ... monster (odd) so you gain x experience points.
        #   - From dp1 (odd), defeat becomes the 2nd, 4th, ... monster (even) so you gain an extra x, i.e., 2*x.
        # Defeating a monster does not alter the order of operations.
        new_dp1 = max(dp1, dp0 + x)   # Transition from even count: defeating gives odd count.
        new_dp0 = max(dp0, dp1 + 2*x)   # Transition from odd count: defeating gives even count.
        dp0, dp1 = new_dp0, new_dp1

    # The answer is the maximum xp accumulated, whether we have an even or odd number of defeats.
    ans = max(dp0, dp1)
    sys.stdout.write(str(ans))

if __name__ == '__main__':
    main()