def main():
    import sys
    data = sys.stdin.read().strip().split()
    K = int(data[0])  # always 1 in this problem
    S = data[1]
    T = data[2]
    
    # Quick checks:
    # 1) If S == T, we are already done.
    if S == T:
        print("Yes")
        return
    
    # 2) Since K=1, if the length difference is greater than 1, it can't be done in one operation.
    len_s = len(S)
    len_t = len(T)
    if abs(len_s - len_t) > 1:
        print("No")
        return
    
    # Function to check if S can be changed to T by exactly one replacement
    def can_replace_once(s, t):
        # Both lengths are the same
        diff_count = 0
        for c1, c2 in zip(s, t):
            if c1 != c2:
                diff_count += 1
                if diff_count > 1:
                    return False
        return diff_count == 1
    
    # Function to check if we can transform S into T by exactly one insertion
    # (equivalently T is S with one extra char somewhere).
    def can_insert_once(s, t):
        # len(t) == len(s)+1
        # We'll try to walk through s and t with at most one mismatch.
        i = 0
        j = 0
        used_insertion = False
        while i < len(s) and j < len(t):
            if s[i] == t[j]:
                i += 1
                j += 1
            else:
                if used_insertion:
                    return False
                used_insertion = True
                j += 1  # Skip one character in t since it's like t had that extra char
        return True  # If there's any leftover in t, that's okay, only up to 1 char
    
    # Function to check if we can transform S into T by exactly one deletion
    # (equivalently T is S minus one char).
    # This is the same as checking if we can transform T into S by exactly one insertion.
    def can_delete_once(s, t):
        return can_insert_once(t, s)
    
    # Check each possible operation:
    # Replacement only works if lengths are equal and they differ in exactly one position
    if len_s == len_t and can_replace_once(S, T):
        print("Yes")
        return
    
    # Insertion only works if T is length(S)+1
    if len_t == len_s + 1 and can_insert_once(S, T):
        print("Yes")
        return
    
    # Deletion only works if T is length(S)-1
    if len_s == len_t + 1 and can_delete_once(S, T):
        print("Yes")
        return
    
    # If none of the above worked, print No
    print("No")

if __name__ == "__main__":
    main()