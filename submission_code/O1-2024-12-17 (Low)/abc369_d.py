def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # dp_even: max experience with an even number of kills so far
    # dp_odd:  max experience with an odd number of kills so far
    dp_even, dp_odd = 0, -float("inf")

    for x in A:
        # If we skip the monster, dp_even and dp_odd stay the same
        # If we defeat the monster:
        #   - from dp_even we go to dp_odd adding x (since next kill is odd),
        #   - from dp_odd  we go to dp_even adding 2*x (since next kill is even).
        new_even = max(dp_even, dp_odd + 2 * x)
        new_odd = max(dp_odd, dp_even + x)
        dp_even, dp_odd = new_even, new_odd

    # The answer is the maximum of dp_even or dp_odd after considering all monsters
    print(max(dp_even, dp_odd))

# Don't forget to call main()
if __name__ == "__main__":
    main()