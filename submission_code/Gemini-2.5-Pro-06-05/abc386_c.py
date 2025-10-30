import sys

def main():
    """
    Reads two strings S and T and an integer K=1 from standard input.
    Determines if S can be transformed into T using at most K operations
    (insert, delete, or replace a single character).
    Prints "Yes" or "No" to standard output.
    """
    
    # It is good practice to read input efficiently for large inputs.
    # K is given on the first line, but is fixed to 1 for this problem.
    # We must read it to consume the line from the input stream.
    try:
        K = int(sys.stdin.readline())
        S = sys.stdin.readline().strip()
        T = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # This handles cases with empty or malformed input,
        # which shouldn't occur based on the problem constraints.
        return

    len_s = len(S)
    len_t = len(T)

    # Case 0: 0 operations are needed (S and T are identical).
    # The number of operations is 0, which is <= K.
    if S == T:
        print("Yes")
        return

    # Since K=1, if the strings are not identical, they must be transformable
    # in exactly one operation. The three types of operations (replace, insert, delete)
    # are mutually exclusive based on the resulting string lengths.

    # Case 1: 1 operation of Replacement.
    # This is only possible if the strings have the same length.
    if len_s == len_t:
        diff_count = 0
        for i in range(len_s):
            if S[i] != T[i]:
                diff_count += 1
        
        # A single replacement is possible if there is exactly one differing character.
        if diff_count == 1:
            print("Yes")
        else:
            print("No")
        return

    # Case 2: 1 operation of Insertion (into S to get T).
    # This is only possible if T is one character longer than S.
    if len_t == len_s + 1:
        # Find the first index where S and T differ.
        i = 0
        while i < len_s and S[i] == T[i]:
            i += 1
        
        # The character T[i] must be the one that was inserted.
        # The remainder of S (from index i) must match the remainder of T (from index i+1).
        # This correctly handles insertion at the beginning, middle, or end.
        if S[i:] == T[i+1:]:
            print("Yes")
        else:
            print("No")
        return

    # Case 3: 1 operation of Deletion (from S to get T).
    # This is only possible if S is one character longer than T.
    if len_s == len_t + 1:
        # This logic is symmetric to the insertion case.
        # Find the first index where S and T differ.
        i = 0
        while i < len_t and S[i] == T[i]:
            i += 1

        # The character S[i] must be the one that was deleted.
        # The remainder of S (from index i+1) must match the remainder of T (from index i).
        if S[i+1:] == T[i:]:
            print("Yes")
        else:
            print("No")
        return
        
    # If none of the above length relationships hold (e.g., length difference > 1),
    # it's impossible to match the strings in at most 1 operation.
    print("No")

if __name__ == "__main__":
    main()