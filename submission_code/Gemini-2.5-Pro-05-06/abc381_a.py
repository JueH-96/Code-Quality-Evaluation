def solve():
    N = int(input())
    S = input()

    # k_val represents the count of '1's before the slash
    # and the count of '2's after the slash in a potential 11/22 string.
    #
    # An 11/22 string, by definition, must have an odd length.
    # Let this length be L. Then L = 2 * k_val + 1.
    # This means k_val = (L - 1) / 2.
    #
    # We use integer division N // 2 to determine k_val based on the input length N.
    #
    # If N (the length of S) is odd, say N = 2m + 1 for some integer m >= 0:
    #   k_val = (2m + 1) // 2 = m.
    #   The length of the generated expected_S will be m (for '1's) + 1 (for '/') + m (for '2's) = 2m + 1.
    #   This length (2m + 1) is equal to N.
    #   In this case, S and expected_S will have the same length. The comparison S == expected_S
    #   will then check if S has the correct characters in the correct positions.
    #
    # If N (the length of S) is even, say N = 2m for some integer m > 0:
    #   k_val = (2m) // 2 = m.
    #   The length of the generated expected_S will be m + 1 + m = 2m + 1.
    #   This length (2m + 1) is N + 1.
    #   In this case, S (length N) and expected_S (length N+1) will have different lengths.
    #   Thus, S == expected_S will be false. This correctly implies that S is not an 11/22 string,
    #   as one of the conditions is that the length must be odd.
    
    k_val = N // 2
    
    # Construct the canonical 11/22 string based on k_val.
    # This string consists of k_val '1's, followed by a '/', followed by k_val '2's.
    expected_S = '1' * k_val + '/' + '2' * k_val
    
    # Check if the input string S matches this canonical form.
    if S == expected_S:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()