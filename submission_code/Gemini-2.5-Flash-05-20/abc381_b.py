import collections

def solve():
    S = input()

    # Condition 1: |T| is even.
    if len(S) % 2 != 0:
        print("No")
        return

    # Condition 2: For each integer i satisfying 1 <= i <= |T|/2,
    # the (2i-1)-th and 2i-th characters of T are equal.
    # In 0-indexed terms, this means S[0]==S[1], S[2]==S[3], and so on.
    # We iterate with a step of 2.
    for i in range(0, len(S), 2):
        if S[i] != S[i+1]:
            print("No")
            return

    # Condition 3: Each character appears in T exactly zero or two times.
    # That is, every character contained in T appears exactly twice in T.
    # We use collections.Counter to count character frequencies.
    char_counts = collections.Counter(S)
    for char, count in char_counts.items():
        # For every character that *is* contained in S, its count must be 2.
        # Characters not contained in S implicitly have a count of 0, satisfying the "zero or two times" rule.
        if count != 2:
            print("No")
            return

    # If all three conditions are satisfied, print Yes.
    print("Yes")

# Call the solve function to run the program
solve()