# YOUR CODE HERE
import sys

def solve():
    # K is given as 1 in this specific problem (sub-problem F, K=1)
    k_val = int(sys.stdin.readline()) # Read K, although it's fixed to 1
    s = sys.stdin.readline().strip()
    t = sys.stdin.readline().strip()

    len_s = len(s)
    len_t = len(t)

    # Case 1: S and T are already identical (0 operations needed)
    if s == t:
        print("Yes")
        return

    # Case 2: Length difference is greater than K (1 for this problem)
    # Impossible to transform S to T in 1 operation if lengths differ by more than 1.
    if abs(len_s - len_t) > 1:
        print("No")
        return

    # Case 3: Calculate common prefix length
    prefix_match_len = 0
    # Loop condition ensures we don't go out of bounds for either string
    while prefix_match_len < min(len_s, len_t) and s[prefix_match_len] == t[prefix_match_len]:
        prefix_match_len += 1

    # Case 4: Calculate common suffix length
    suffix_match_len = 0
    # The condition `suffix_match_len < min(len_s, len_t) - prefix_match_len` ensures
    # that the suffix comparison does not overlap with the prefix comparison.
    # It accounts for cases where the prefix match covers a significant portion,
    # or even all, of one of the strings.
    while suffix_match_len < min(len_s, len_t) - prefix_match_len and \
          s[len_s - 1 - suffix_match_len] == t[len_t - 1 - suffix_match_len]:
        suffix_match_len += 1

    # Total characters that matched from both ends without overlapping
    total_matched_len = prefix_match_len + suffix_match_len

    # Based on the length relationship and total matched characters:
    if len_s == len_t:
        # If lengths are equal, it must be a single character replacement.
        # This means exactly one character should differ.
        # The number of non-matching characters is len_s - total_matched_len.
        # If this count is 1, then it's a "Yes".
        if total_matched_len == len_s - 1:
            print("Yes")
        else:
            print("No")
    elif len_s == len_t + 1: # S is longer than T by 1 (possible deletion from S)
        # If T can be formed by deleting one character from S, then all characters of T
        # must be covered by the matched parts of S.
        # This means the length of T should be equal to the total matched length.
        if total_matched_len == len_t:
            print("Yes")
        else:
            print("No")
    elif len_s == len_t - 1: # S is shorter than T by 1 (possible insertion into S)
        # If T can be formed by inserting one character into S, then all characters of S
        # must be covered by the matched parts of T.
        # This means the length of S should be equal to the total matched length.
        if total_matched_len == len_s:
            print("Yes")
        else:
            print("No")
    # No other cases are possible due to the `abs(len_s - len_t) > 1` check.
    # This final else branch is theoretically unreachable if the above conditions cover all valid length differences within K.
    else:
        print("No")

# Call the solve function to execute the logic
solve()