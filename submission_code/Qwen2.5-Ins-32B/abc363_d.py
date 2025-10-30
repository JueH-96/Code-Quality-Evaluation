def is_palindrome(num):
    return str(num) == str(num)[::-1]

def find_nth_palindrome(n):
    if n == 1:
        return 0
    count = 1  # Starting with 0 as the first palindrome
    num = 1
    while count < n:
        if is_palindrome(num):
            count += 1
        if count == n:
            return num
        num += 1

def generate_palindrome(n):
    if n <= 10:
        return n - 1
    length = 1
    while n > 9 * 10**(length - 1):
        n -= 9 * 10**(length - 1)
        length += 1
    half = 10**(length // 2 - 1) + (n - 1) // 10**(length // 2)
    if length % 2 == 0:
        return int(str(half) + str(half)[::-1])
    else:
        return int(str(half) + str(half)[-2::-1])

n = int(input())
print(generate_palindrome(n))