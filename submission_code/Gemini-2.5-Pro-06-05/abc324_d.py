import sys

def solve():
    """
    Solves the problem by iterating through square numbers and checking if they
    can be formed by a permutation of the input string's digits.
    """
    # Read input from stdin for N and S.
    try:
        N = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Handle cases with empty or malformed input.
        return

    # Count the frequency of each digit in the input string S.
    # This serves as a canonical representation of the multiset of digits available.
    s_counts = [0] * 10
    for digit in S:
        s_counts[int(digit)] += 1

    # A number formed by N digits is always less than 10^N.
    # So, we only need to check squares up to 10^N - 1.
    # The loop for k goes up to sqrt(10^N - 1).
    limit = 10**N
    count = 0
    k = 0
    while True:
        # Generate the next square number.
        sq = k * k

        # If the square is too large, we can stop, as subsequent squares will also be too large.
        if sq >= limit:
            break

        # Convert the square to a string to analyze its digits.
        sq_str = str(sq)

        # To be formed by a permutation of S, a number must have the same
        # number of digits (N), including any necessary leading zeros.
        # We check if the multiset of digits of the square (with padding)
        # matches the multiset of digits of S.
        
        # Count the digits in the current square.
        sq_counts = [0] * 10
        for digit in sq_str:
            sq_counts[int(digit)] += 1
        
        # Add the count for required leading zeros to make the length N.
        sq_counts[0] += N - len(sq_str)

        # If the digit counts match, we have found a valid square number.
        if sq_counts == s_counts:
            count += 1
            
        k += 1

    # Print the final answer to stdout.
    print(count)

solve()