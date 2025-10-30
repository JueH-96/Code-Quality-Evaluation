def solve():
    import sys
    S = sys.stdin.readline().strip()
    
    n = len(S)
    max_len = 1  # Since every single character is a palindrome of length 1
    
    for start in range(n):
        for end in range(start, n):
            # Check if S[start:end+1] is palindrome
            substring = S[start:end+1]
            if substring == substring[::-1]:
                max_len = max(max_len, end - start + 1)
    
    print(max_len)

def main():
    solve()

if __name__ == "__main__":
    main()