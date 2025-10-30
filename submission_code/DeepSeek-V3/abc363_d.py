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
    half = str(10 ** (half_length - 1) + N - 1)
    if length % 2 == 0:
        return int(half + half[::-1])
    else:
        return int(half + half[:-1][::-1])

N = int(input())
print(find_nth_palindrome(N))