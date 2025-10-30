def main():
    import sys

    S = sys.stdin.readline().strip()
    T = sys.stdin.readline().strip()
    
    def is_subsequence(s, pattern):
        """Return True if 'pattern' is a subsequence of 's'."""
        i = 0
        for c in s:
            if i < len(pattern) and c == pattern[i]:
                i += 1
            if i == len(pattern):
                return True
        return False
    
    # Method 1: Check if T (lowercased) is a length 3 subsequence of S
    method1 = is_subsequence(S, T.lower())
    
    # Method 2: Check if T ends with 'X' and the first two letters (lowercased) form a subsequence of S
    method2 = (T[-1] == 'X') and is_subsequence(S, T[:2].lower())
    
    # If either condition is satisfied, print Yes; otherwise, print No
    if method1 or method2:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()