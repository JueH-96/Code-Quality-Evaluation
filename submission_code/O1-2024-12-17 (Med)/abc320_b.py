def main():
    import sys

    S = sys.stdin.readline().strip()
    n = len(S)
    max_len = 1  # Since there's always at least a length-1 palindrome

    for i in range(n):
        for j in range(i, n):
            # Check if the substring S[i:j+1] is a palindrome
            substring = S[i:j+1]
            if substring == substring[::-1]:
                max_len = max(max_len, j - i + 1)
    print(max_len)

# Call main function
if __name__ == "__main__":
    main()