# YOUR CODE HERE
def is_palindrome(s):
    return s == s[::-1]

def max_palindrome_length(s):
    max_length = 1
    n = len(s)
    for i in range(n):
        for j in range(i + 1, n + 1):
            if is_palindrome(s[i:j]):
                max_length = max(max_length, j - i)
    return max_length

if __name__ == "__main__":
    S = input().strip()
    print(max_palindrome_length(S))