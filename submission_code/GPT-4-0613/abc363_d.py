def nth_palindrome(n):
    if n < 10:
        return n - 1
    n -= 10
    digits = 2
    while True:
        first_half_count = 10 * (10 ** (digits // 2 - 1))
        if n < first_half_count:
            break
        n -= first_half_count
        digits += 1
    first_half = str(10 ** (digits // 2 - 1) + n // (digits % 2 + 1))
    second_half = first_half[:-digits % 2][::-1]
    return int(first_half + second_half)

n = int(input().strip())
print(nth_palindrome(n+1))