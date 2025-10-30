def main():
    S = input().strip()
    
    def is_all_char(substr, ch):
        return all(c == ch for c in substr)
    
    n = len(S)
    # Try all possible splits i, j where:
    #   S_A = S[0:i] (all A's)
    #   S_B = S[i:j] (all B's)
    #   S_C = S[j:]  (all C's)
    for i in range(n+1):
        for j in range(i, n+1):
            S_A = S[:i]
            S_B = S[i:j]
            S_C = S[j:]
            if is_all_char(S_A, 'A') and is_all_char(S_B, 'B') and is_all_char(S_C, 'C'):
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()