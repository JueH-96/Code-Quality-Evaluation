def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]
    
    def is_palindrome(s):
        return s == s[::-1]
    
    for i in range(N):
        for j in range(N):
            if i != j:
                if is_palindrome(S[i] + S[j]):
                    print("Yes")
                    return
    
    print("No")

if __name__ == "__main__":
    main()