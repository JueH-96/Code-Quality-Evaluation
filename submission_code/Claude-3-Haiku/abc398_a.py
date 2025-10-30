import sys

def find_palindrome(N):
    if N % 2 == 0:
        # N is even
        return "-" * (N//2 - 1) + "==" + "-" * (N//2 - 1)
    else:
        # N is odd
        return "-" * (N//2) + "=" + "-" * (N//2)

N = int(sys.stdin.readline().strip())
print(find_palindrome(N))