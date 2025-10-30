def solve():
    import sys
    
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    # Helper function to check if t is a subsequence of s.
    def is_subsequence(s, t):
        # Attempt to match characters of t in order within s.
        # t should be lowercase to match the lowercase s.
        idx_t = 0
        len_t = len(t)
        for char in s:
            if char == t[idx_t]:
                idx_t += 1
                if idx_t == len_t:
                    return True
        return False
    
    # 1) Check if T (converted to lowercase) is a 3-letter subsequence of S.
    if is_subsequence(S, T.lower()):
        print("Yes")
        return
    
    # 2) Check if T is of the form XX + 'X' and the first two letters (lowercase) are a subsequence of S.
    if T[2] == 'X':
        if is_subsequence(S, T[:2].lower()):
            print("Yes")
            return
    
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()