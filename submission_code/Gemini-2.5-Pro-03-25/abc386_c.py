# YOUR CODE HERE
import sys

def solve():
    """
    Reads K, S, and T from standard input and determines if S can be transformed
    into T using at most K=1 edit operation (insert, delete, or replace).
    Prints "Yes" or "No" to standard output.

    The problem statement guarantees K=1 for this sub-problem.
    The operations allowed are:
    1. Insert a character into S.
    2. Delete a character from S.
    3. Replace a character in S.

    We need to check if S can be made identical to T using 0 or 1 operation.
    """

    # Read K from the first line. Although K=1 is guaranteed, we still need to read the line.
    # We don't actually need the value of K for the logic since it's fixed at 1.
    sys.stdin.readline()

    # Read string S from the second line and remove leading/trailing whitespace (like newline)
    s = sys.stdin.readline().strip()
    # Read string T from the third line and remove leading/trailing whitespace
    t = sys.stdin.readline().strip()

    len_s = len(s)
    len_t = len(t)

    # --- Case 0: Check for 0 operations ---
    # If S is already identical to T, 0 operations are needed, which is <= K=1.
    if s == t:
        print("Yes")
        return

    # If S != T, we need to check if exactly 1 operation suffices.

    # --- Case 1: Check for 1 replacement operation ---
    # A single replacement is possible only if S and T have the same length
    # and differ by exactly one character.
    if len_s == len_t:
        diff_count = 0
        # Iterate through the strings and count the number of positions where characters differ
        for i in range(len_s):
            if s[i] != t[i]:
                diff_count += 1
            # Optimization: If we find more than one difference,
            # it's impossible to achieve the transformation with a single replacement.
            # We can stop checking early.
            if diff_count > 1:
                break

        # If exactly one difference was found after checking the entire string (or stopping early)
        if diff_count == 1:
            print("Yes")
            return

    # --- Case 2: Check for 1 insertion operation (transforming S to T) ---
    # A single insertion into S to get T is possible only if T has exactly one more character than S.
    elif len_t == len_s + 1:
        # We need to find if T can be formed by inserting one character into S.
        # This means S should be equal to T with one character removed.
        # Find the first index where S and T differ.
        i = 0
        # Compare characters up to the length of the shorter string (S).
        while i < len_s and s[i] == t[i]:
            i += 1

        # At this point, 'i' is the index of the first difference, or i == len_s if S is a prefix of T.
        # The character t[i] must be the character that was inserted.
        # We check if the remainder of S (s[i:]) matches the remainder of T after skipping t[i] (t[i+1:]).
        # This covers insertion at the beginning (i=0), middle, and end (i=len_s).
        # Example: S="abc", T="axbc". i=1. s[1:]="bc", t[1+1:]=t[2:]="bc". Match.
        # Example: S="abc", T="abcd". i=3. s[3:]="", t[3+1:]=t[4:]="". Match.
        if s[i:] == t[i+1:]:
            print("Yes")
            return

    # --- Case 3: Check for 1 deletion operation (transforming S to T) ---
    # A single deletion from S to get T is possible only if S has exactly one more character than T.
    elif len_s == len_t + 1:
        # We need to find if T can be formed by deleting one character from S.
        # This means T should be equal to S with one character removed.
        # Find the first index where S and T differ.
        i = 0
        # Compare characters up to the length of the shorter string (T).
        while i < len_t and s[i] == t[i]:
            i += 1

        # At this point, 'i' is the index of the first difference, or i == len_t if T is a prefix of S.
        # The character s[i] must be the character that was deleted.
        # We check if the remainder of T (t[i:]) matches the remainder of S after skipping s[i] (s[i+1:]).
        # This covers deletion at the beginning (i=0), middle, and end (i=len_t).
        # Example: S="axbc", T="abc". i=1. s[1+1:]=s[2:]="bc", t[1:]="bc". Match.
        # Example: S="abcd", T="abc". i=3. s[3+1:]=s[4:]="", t[3:]="". Match.
        if s[i+1:] == t[i:]:
            print("Yes")
            return

    # If none of the above conditions (0 operations or exactly 1 operation of any type) were met,
    # it's impossible to transform S to T in at most 1 operation.
    print("No")

# Call the solve function to execute the logic when the script runs
solve()