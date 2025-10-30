def find_nth_palindrome(N):
    if N == 1:
        return 0
    N -= 1
    length = 1
    while True:
        half_length = (length + 1) // 2
        count = 9 * (10 ** (half_length - 1))
        if N <= count:
            break
        N -= count
        length += 1
    half_length = (length + 1) // 2
    first_half = 10 ** (half_length - 1) + N - 1
    first_half_str = str(first_half)
    if length % 2 == 0:
        palindrome = first_half_str + first_half_str[::-1]
    else:
        palindrome = first_half_str + first_half_str[:-1][::-1]
    return int(palindrome)

N = int(input())
print(find_nth_palindrome(N))