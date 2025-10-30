def find_nth_palindrome(n):
    if n == 1:
        return 0
    n -= 1
    digits = 1
    while n >= 10 ** (digits // 2):
        n -= 10 ** (digits // 2)
        digits += 1
    half = str(n)
    half = '0' * (digits // 2 - len(half)) + half
    if digits % 2 == 0:
        return int(half + half[::-1])
    else:
        return int(half + half[-2::-1])

n = int(input())
print(find_nth_palindrome(n))