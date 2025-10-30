def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    
    # A helper function to check if a given string is a palindrome
    def is_palindrome(substr):
        return substr == substr[::-1]
    
    # We'll check substrings from largest to smallest length
    for length in range(n, 0, -1):
        for start in range(n - length + 1):
            # Extract the substring
            substr = S[start:start+length]
            if is_palindrome(substr):
                print(length)
                return

# Do not forget to call main()
main()