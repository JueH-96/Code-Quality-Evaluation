def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    S = [input().strip() for _ in range(N)]
    
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            combined = S[i] + S[j]
            # Check if 'combined' is a palindrome
            if combined == combined[::-1]:
                print("Yes")
                return
    
    print("No")

if __name__ == "__main__":
    main()