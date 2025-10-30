S = input()
T = input()

if S == T:
    print(0)
else:
    len_s = len(S)
    len_t = len(T)
    min_len = min(len_s, len_t)

    # Iterate through the common part of the strings
    for i in range(min_len):
        if S[i] != T[i]:
            # Found the first differing character in the common part
            print(i + 1)  # Output 1-based index
            break  # Exit loop, the 'else' block below will not run
    else:
        # This 'else' block executes if the 'for' loop completed without a 'break'.
        # This means S and T are identical up to min_len characters.
        # Since we know S != T (from the initial 'if' condition),
        # their lengths must be different.
        # The first differing position is therefore min_len + 1.
        print(min_len + 1)