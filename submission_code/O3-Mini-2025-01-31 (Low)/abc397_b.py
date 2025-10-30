def main():
    import sys
    S = sys.stdin.read().strip()
    if not S:
        return
    n = len(S)
    
    # We want to insert the minimum characters so that S becomes a subsequence
    # of a valid pattern string where the pattern is "i", "o", "i", "o", ... of even length.
    # The answer is (L - len(S)), where L is the smallest even number such that S is a subsequence
    # of the pattern of length L.
    #
    # Brute force by trying even lengths from n (or n+1 if n is odd) upward is efficient enough
    # because n is at most 100.
    
    # Start L from n, but if n is odd the valid pattern requires even length so L = n+1 in that case.
    start = n if n % 2 == 0 else n + 1
    
    # We try even values of L until we find one that contains S as a subsequence.
    for L in range(start, 301, 2):
        j = 0  # index to track progress in S
        # Simulate stepping through the valid pattern of length L.
        # In a valid pattern the odd-indexed (1st, 3rd, ...) characters are 'i' and
        # the even-indexed (2nd, 4th, ...) characters are 'o'. (Note: indices 0,2,4... correspond to these positions)
        for p in range(L):
            expected_char = 'i' if p % 2 == 0 else 'o'
            if j < n and S[j] == expected_char:
                j += 1
        # If we have matched the entire S as a subsequence, then we have found a valid L
        if j == n:
            # The number of insertions is the difference between the valid length and the original length.
            print(L - n)
            return

if __name__ == '__main__':
    main()