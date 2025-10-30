# YOUR CODE HERE
n = int(input())

if n <= 10:
    print(n - 1)
else:
    n -= 10
    digits = 2
    
    while True:
        count = 9 * (10 ** ((digits - 1) // 2))
        if n <= count:
            break
        n -= count
        digits += 1
    
    half_len = (digits + 1) // 2
    first_half = 10 ** (half_len - 1) + n - 1
    first_half_str = str(first_half)
    
    if digits % 2 == 0:
        palindrome = first_half_str + first_half_str[::-1]
    else:
        palindrome = first_half_str + first_half_str[-2::-1]
    
    print(palindrome)