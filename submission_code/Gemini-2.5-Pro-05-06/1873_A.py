def solve_one_case():
    s = input()
    # The target string is "abc".
    # We need to check if the given string 's' can become "abc"
    # in at most one swap.

    # This is true if:
    # 1. 's' is already "abc" (0 swaps).
    #    This means s[0]=='a', s[1]=='b', and s[2]=='c'.
    # 2. 's' can become "abc" in 1 swap.
    #    This means 's' has exactly one character in the correct
    #    target position, and the other two are swapped.
    #    For example, if s = "acb":
    #      s[0]=='a' (correct)
    #      s[1]=='c' (should be 'b')
    #      s[2]=='b' (should be 'c')
    #    Swapping s[1] and s[2] gives "abc".

    # Combined condition:
    # If at least one character is in its correct final position ("abc"),
    # then it's possible with at most one swap.
    #   - If all three are correct (s == "abc"), 0 swaps.
    #   - If one is correct (e.g., s[0]=='a' but s[1]!='b', s[2]!='c'),
    #     then s must be like "acb". The other two ('c' and 'b') must be
    #     in each other's places. One swap (s[1] with s[2]) fixes it.
    # If no character is in its correct final position (e.g., "bca" or "cab"),
    # then it's a 3-cycle, which requires 2 swaps.

    # Check if s[0] is 'a', or s[1] is 'b', or s[2] is 'c'.
    if s[0] == 'a' or s[1] == 'b' or s[2] == 'c':
        print("YES")
    else:
        print("NO")

# Read the number of test cases
num_test_cases = int(input())

# Iterate through each test case
for _ in range(num_test_cases):
    solve_one_case()