# YOUR CODE HERE
import sys

def is_palindrome(s):
    return s == s[::-1]

def max_palindrome_length(s):
    n = len(s)
    max_length = 1
    for i in range(n):
        for j in range(i + 1, n + 1):
            if is_palindrome(s[i:j]):
                max_length = max(max_length, j - i)
    return max_length

def main():
    input = sys.stdin.read().strip()
    print(max_palindrome_length(input))

if __name__ == "__main__":
    main()