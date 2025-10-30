def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    S = data[1:]
    
    # A quick helper function to check if a string is a palindrome
    def is_palindrome(s):
        return s == s[::-1]
    
    # We just need to check every pair (i, j) with i != j
    for i in range(N):
        for j in range(N):
            if i != j:
                # Concatenate S[i] + S[j]
                if is_palindrome(S[i] + S[j]):
                    print("Yes")
                    return
    
    # If no such pair is found, print No
    print("No")

def main():
    solve()

if __name__ == "__main__":
    main()