import sys

def solve():
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()

    len_S = len(S)
    len_T = len(T)

    # Determine the length of the shorter string, as we only need to compare up to this point
    min_len = min(len_S, len_T)

    # Iterate through the strings character by character up to the length of the shorter string
    for i in range(min_len):
        # If a difference is found, print the 1-based index and exit
        if S[i] != T[i]:
            print(i + 1)  # Convert 0-based index to 1-based position
            return

    # If the loop completes, it means all characters in the common prefix are identical.
    # Now, check the lengths of the strings.
    if len_S == len_T:
        # If lengths are also equal, the strings are identical
        print(0)
    else:
        # If lengths are different, one string is a prefix of the other.
        # The first differing position is the length of the shorter string plus one.
        print(min_len + 1)

# Call the solve function to execute the logic
solve()