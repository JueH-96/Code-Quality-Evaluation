# YOUR CODE HERE
def main():
    """
    This program solves the Vanya and Vova game by determining the winner
    based on the initial number n.
    """
    try:
        # Read the number of test cases.
        num_test_cases = int(input())
    except (ValueError, EOFError):
        # Handle cases where input is not a valid integer or stream is empty.
        return

    # Iterate through each test case.
    for _ in range(num_test_cases):
        try:
            n = int(input())
            
            # The winner is determined by the remainder of n when divided by 3.
            # If n is divisible by 3, the second player has a winning strategy.
            # Otherwise, the first player can win on their first move.
            if n % 3 == 0:
                print("Second")
            else:
                print("First")
        except (ValueError, EOFError):
            # In case of malformed input for n, we can simply continue
            # as per standard competitive programming practices where
            # test data is usually well-formed.
            continue

if __name__ == "__main__":
    main()