def main():
    import sys

    data = sys.stdin.read().strip().split()
    X, Y, Z = map(int, data[:3])
    S = data[3]

    # dp0 = minimum cost to type prefix with CapsLock OFF
    # dp1 = minimum cost to type prefix with CapsLock ON
    dp0, dp1 = 0, Z  # starting with OFF costs 0, with ON costs Z (one toggle)

    for c in S:
        if c == 'a':
            # If CapsLock is OFF and we press 'a', we get 'a' (cost X).
            # If CapsLock is ON and we press Shift+'a', we get 'a' (cost Y).
            cost_if_off = X
            cost_if_on = Y
        else:  # c == 'A'
            # If CapsLock is OFF and we press Shift+'a', we get 'A' (cost Y).
            # If CapsLock is ON and we press 'a', we get 'A' (cost X).
            cost_if_off = Y
            cost_if_on = X

        new_dp0 = min(dp0 + cost_if_off,  dp1 + Z + cost_if_off)
        new_dp1 = min(dp1 + cost_if_on,   dp0 + Z + cost_if_on)
        dp0, dp1 = new_dp0, new_dp1

    print(min(dp0, dp1))

# Call the main function
if __name__ == "__main__":
    main()