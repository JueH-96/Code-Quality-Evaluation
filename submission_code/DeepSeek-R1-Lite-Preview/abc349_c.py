def is_subsequence(sub, s):
    it = iter(s)
    return all(c in it for c in sub)

def main():
    import sys
    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    t_lower = T.lower()
    
    # Method 1: Check for subsequence of length 3
    if is_subsequence(t_lower, S):
        print("Yes")
        return
    
    # Method 2: Check for subsequence of length 2 plus 'X'
    if T[2] == 'X' and is_subsequence(t_lower[:2], S):
        print("Yes")
        return
    
    print("No")

if __name__ == "__main__":
    main()