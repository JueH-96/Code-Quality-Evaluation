def main():
    import sys

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    X = list(map(int, data[1:]))

    # The idea (which can be proven) is that each "beneficial" 4-piece reflection
    # can be performed in some sequence without preventing other beneficial
    # reflections.  A reflection is beneficial if it decreases the total sum of
    # all coordinates, i.e. the quantity:
    #
    #   2 * (X_i + X_{i+3} - X_{i+1} - X_{i+2}) < 0
    #
    # for the quadruple (X_i, X_{i+1}, X_{i+2}, X_{i+3}) in ascending order.
    #
    # Therefore, the minimal possible sum equals the original sum of X
    # plus (actually minus) the total of all negative terms of the form
    #  2*(X_i + X_{i+3} - X_{i+1} - X_{i+2}),
    # taken over i=0..(N-4) in 0-based indexing (or i=1..(N-3) in the problem's 1-based indexing).

    total_sum = sum(X)
    adjustment = 0

    # Check each quadruple in the original, sorted array
    for i in range(N - 3):
        delta = 2 * (X[i] + X[i+3] - X[i+1] - X[i+2])
        if delta < 0:  # beneficial reflection
            adjustment += delta

    print(total_sum + adjustment)

# Do not forget to call main!
if __name__ == "__main__":
    main()