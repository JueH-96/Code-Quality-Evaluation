def main():
    import sys
    
    input_data = sys.stdin.read().strip().split()
    # K is guaranteed to be 1 in this sub-problem
    K = int(input_data[0])
    S = input_data[1]
    T = input_data[2]
    
    # Since K=1, we only need to check if S and T are "at most one edit" away.
    # An edit can be: insert one char, delete one char, or replace one char.
    # We'll implement a function to check if two strings are at most one edit apart.

    if K != 1:
        # If for any reason K != 1, the problem statements says K=1, so we just handle that case.
        print("No")
        return
    
    # Quick check: if S == T, 0 operations are needed
    if S == T:
        print("Yes")
        return
    
    # If length difference is greater than 1, it's impossible with only one operation
    if abs(len(S) - len(T)) > 1:
        print("No")
        return
    
    # Define a helper to check if two strings of the same length differ in at most 1 position
    def one_replace_away(a, b):
        # a, b have same length
        diff_count = 0
        for x, y in zip(a, b):
            if x != y:
                diff_count += 1
                if diff_count > 1:
                    return False
        return True
    
    # Define a helper to check if we can make strings of length differing by exactly 1 identical with one insert/delete
    # We try to see if there's a position where a single skip in the longer string aligns it with the shorter one
    def one_insert_away(shorter, longer):
        # len(longer) == len(shorter) + 1
        i = j = 0
        used_skip = False
        while i < len(shorter) and j < len(longer):
            if shorter[i] == longer[j]:
                i += 1
                j += 1
            else:
                if used_skip: 
                    return False
                used_skip = True
                j += 1
        # We can skip the last char in longer if needed (when we haven't used skip),
        # or if we've already matched everything but there's still 1 char at the end in longer.
        return True
    
    # Now check the cases
    len_s = len(S)
    len_t = len(T)
    if len_s == len_t:
        # Possibly fix by 0 or 1 replace
        if one_replace_away(S, T):
            print("Yes")
        else:
            print("No")
    elif len_s + 1 == len_t:
        # Possibly fix by inserting 1 char in S (or effectively skipping in T)
        if one_insert_away(S, T):
            print("Yes")
        else:
            print("No")
    elif len_s == len_t + 1:
        # Possibly fix by deleting 1 char from S (or effectively skipping in S)
        if one_insert_away(T, S):
            print("Yes")
        else:
            print("No")
    else:
        # This case shouldn't happen, because we handled abs difference > 1 above.
        print("No")

# Do not forget to call main()
if __name__ == "__main__":
    main()