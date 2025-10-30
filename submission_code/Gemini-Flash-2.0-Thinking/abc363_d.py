def solve():
    n = int(input())

    if n <= 10:
        print(n - 1)
        return

    count = 10
    digits = 2
    while True:
        num_palindromes = 9 * (10 ** ((digits - 1) // 2))
        if n <= count + num_palindromes:
            break
        count += num_palindromes
        digits += 1

    offset = 0
    for d in range(1, digits):
        if d == 1:
            offset += 10
        else:
            offset += 9 * (10 ** ((d - 1) // 2))

    k = n - offset

    if digits % 2 == 1:
        m = digits // 2
        start = 10**m
        first_half_int = start + k - 1
        first_half_str = str(first_half_int)
        print(first_half_str + first_half_str[:-1][::-1])
    else:
        m = digits // 2
        start = 10**(m - 1)
        first_half_int = start + k - 1
        first_half_str = str(first_half_int)
        print(first_half_str + first_half_str[::-1])