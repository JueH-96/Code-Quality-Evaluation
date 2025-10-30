# YOUR CODE HERE
n = int(input())
if n <= 9:
    print(n - 1)
else:
    n -= 9
    length = 1
    while True:
        count = 9 * (10 ** (length - 1))
        if n <= count:
            break
        n -= count
        length += 1
    num_str = str(10 ** (length - 1) + n - 1)
    palindrome_str = num_str + num_str[:-1][::-1]
    print(int(palindrome_str))