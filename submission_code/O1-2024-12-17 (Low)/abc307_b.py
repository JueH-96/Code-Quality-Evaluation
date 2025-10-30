def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    strings = data[1:]
    
    # A helper function to check palindrome
    def is_palindrome(s):
        return s == s[::-1]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                # Concatenate and check palindrome
                if is_palindrome(strings[i] + strings[j]):
                    print("Yes")
                    return
    print("No")

if __name__ == "__main__":
    main()