# YOUR CODE HERE
import sys

def find_palindrome(N):
    def is_palindrome(s):
        return s == s[::-1]

    def construct_palindrome(N):
        if N < 10:
            return str(N)
        
        for i in range(1, int(N**0.5) + 1):
            if N % i == 0:
                j = N // i
                if str(i) == str(i)[::-1] and str(j) == str(j)[::-1]:
                    return f"{i}*{j}" if i != j else str(i)
                if str(i) == str(i)[::-1]:
                    mid = construct_palindrome(j)
                    if mid != -1:
                        return f"{i}*{mid}*{i}"
                if str(j) == str(j)[::-1]:
                    mid = construct_palindrome(i)
                    if mid != -1:
                        return f"{j}*{mid}*{j}"
        
        return -1

    result = construct_palindrome(N)
    if result == -1 or not is_palindrome(result.replace('*', '')):
        return -1
    return result

N = int(sys.stdin.read().strip())
print(find_palindrome(N))