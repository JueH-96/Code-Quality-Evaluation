import sys

def solve():
    """
    Reads a string from standard input, calculates the number of palindromic triples,
    and prints the result to standard output.
    """
    S = sys.stdin.readline().strip()
    n = len(S)

    if n < 3:
        print(0)
        return

    # counts[c] = number of occurrences of character c seen so far
    # index_sums[c] = sum of indices of occurrences of character c seen so far
    counts = [0] * 26
    index_sums = [0] * 26
    total_triples = 0

    for k, char in enumerate(S):
        char_idx = ord(char) - ord('A')

        # Get the count (m) and sum of indices (s) for the current character
        # from the occurrences before the current index k.
        m = counts[char_idx]
        s = index_sums[char_idx]

        # For a fixed k, the contribution to the total count is the sum of (k - i - 1)
        # over all i < k where S[i] == S[k]. This sum is equal to m*k - s - m.
        if m > 0:
            total_triples += m * k - s - m
        
        # Update the state with the current character's information.
        counts[char_idx] += 1
        index_sums[char_idx] += k

    print(total_triples)

solve()