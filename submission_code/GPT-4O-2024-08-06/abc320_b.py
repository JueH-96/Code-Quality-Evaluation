# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

def max_palindrome_length(S):
    n = len(S)
    max_length = 1  # At least every single character is a palindrome

    for start in range(n):
        for end in range(start + 1, n + 1):
            substring = S[start:end]
            if is_palindrome(substring):
                max_length = max(max_length, end - start)

    return max_length

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    S = input().strip()
    print(max_palindrome_length(S))