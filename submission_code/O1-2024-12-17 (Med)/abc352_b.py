def main():
    import sys
    
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Two pointers:
    # i points to characters in S, j points to characters in T.
    i, j = 0, 0
    len_s, len_t = len(S), len(T)
    
    # We'll record the positions of T where the correct character was typed.
    positions = []
    
    while i < len_s and j < len_t:
        if S[i] == T[j]:
            # We found a correct match for S[i] in T[j].
            positions.append(j + 1)  # +1 for 1-based index
            i += 1
        j += 1
    
    # Print the positions in ascending order (they are already in ascending order).
    print(" ".join(map(str, positions)))

if __name__ == "__main__":
    main()