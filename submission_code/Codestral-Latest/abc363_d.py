# YOUR CODE HERE
def is_palindrome(n):
    return str(n) == str(n)[::-1]

def find_nth_palindrome(N):
    count = 0
    num = 0
    while count < N:
        if is_palindrome(num):
            count += 1
        num += 1
    return num - 1

# Read input
N = int(input().strip())

# Find and print the N-th smallest palindrome number
result = find_nth_palindrome(N)
print(result)