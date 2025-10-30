def sum_up_to(m):
    if m == 1:
        return 10
    elif m % 2 == 0:
        k = m // 2
        return 2 * (10 ** k) - 1
    else:
        k = (m - 1) // 2
        return 11 * (10 ** k) - 1

n = int(input())
if n == 1:
    print(0)
else:
    m = 1
    while True:
        current_sum = sum_up_to(m)
        if current_sum >= n:
            break
        m += 1
    prev_sum = sum_up_to(m - 1)
    pos = n - prev_sum
    if m == 1:
        print(pos - 1)
    else:
        if m % 2 == 0:
            k = m // 2
            first_half = 10 ** (k - 1) + (pos - 1)
            palindrome = first_half * (10 ** k) + int(str(first_half)[::-1])
        else:
            k = (m - 1) // 2
            first_part = 10 ** k + (pos - 1)
            s = str(first_part)
            palindrome_str = s + s[:-1][::-1]
            palindrome = int(palindrome_str)
        print(palindrome)