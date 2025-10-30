n = int(input())
if n == 1:
    print(0)
else:
    k = n - 2
    cumulative = 0
    current_length = 1
    while True:
        if current_length == 1:
            count = 9
        else:
            if current_length % 2 == 0:
                half = current_length // 2
                count = 9 * (10 ** (half - 1))
            else:
                half = (current_length - 1) // 2
                count = 9 * (10 ** half)
        if cumulative + count > k:
            break
        else:
            cumulative += count
            current_length += 1
    if current_length % 2 == 0:
        half = current_length // 2
        s = 10 ** (half - 1) + (k - cumulative)
        palindrome = int(str(s) + str(s)[::-1])
    else:
        half = (current_length - 1) // 2
        s = 10 ** half + (k - cumulative)
        s_str = str(s)
        palindrome = int(s_str + s_str[:-1][::-1])
    print(palindrome)