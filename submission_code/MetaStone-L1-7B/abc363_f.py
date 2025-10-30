def is_palindrome(s):
    return s == s[::-1]

def find_palindrome_string(N):
    s = str(N)
    if is_palindrome(s):
        return s
    # Try k=2
    max_a = int(N**0.5)
    for a in range(1, max_a + 1):
        if N % a == 0:
            b = N // a
            if is_palindrome(str(b)):
                s_candidate = f"{a}*{b}"
                if is_palindrome(s_candidate):
                    return s_candidate
    return "-1"

# Read input
N = int(input())
result = find_palindrome_string(N)
print(result)