# YOUR CODE HERE
def nth_palindrome(n):
    if n <= 9:
        return n - 1
    
    n -= 9
    digits = 1
    while n > 9 * 10**digits:
        n -= 9 * 10**digits
        digits += 1
    
    left = 10**digits + (n - 1) // 2
    if digits % 2 == 0:
        return int(str(left) + str(left)[::-1])
    else:
        return int(str(left) + str(left)[-2::-1])

n = int(input())
print(nth_palindrome(n))