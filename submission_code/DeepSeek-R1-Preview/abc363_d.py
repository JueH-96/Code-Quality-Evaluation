def find_nth_palindrome():
    N = int(input().strip())
    if N <= 10:
        print(N - 1)
        return
    
    current = N - 10
    l = 2
    
    while True:
        if l % 2 == 0:
            k = l // 2
            count = 9 * (10 ** (k - 1))
        else:
            k = (l - 1) // 2
            count = 9 * (10 ** k)
        
        if current <= count:
            break
        current -= count
        l += 1
    
    if l % 2 == 0:
        k = l // 2
        x = 10 ** (k - 1) + (current - 1)
        s = str(x)
        res = s + s[::-1]
    else:
        k = (l - 1) // 2
        x_index = (current - 1) // 10
        d = (current - 1) % 10
        x = 10 ** (k - 1) + x_index
        s = str(x)
        res = s + str(d) + s[::-1]
    
    print(res)

find_nth_palindrome()