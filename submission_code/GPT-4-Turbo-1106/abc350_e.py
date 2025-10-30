def expected_cost(N, A, X, Y):
    # Base case: if N is already 0, no cost is needed.
    if N == 0:
        return 0.0

    # Calculate the expected cost of rolling the die.
    expected_roll_cost = sum(expected_cost(N // b, A, X, Y) for b in range(1, 7)) / 6.0 + Y

    # Calculate the cost of using the guaranteed division.
    guaranteed_division_cost = expected_cost(N // A, A, X, Y) + X

    # Return the minimum of the two expected costs.
    return min(expected_roll_cost, guaranteed_division_cost)

def main():
    N, A, X, Y = map(int, input().split())
    print("{:.10f}".format(expected_cost(N, A, X, Y)))

if __name__ == "__main__":
    main()