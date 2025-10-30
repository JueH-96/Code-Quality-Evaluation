def find_nth_palindrome():
    import sys
    N = int(sys.stdin.readline().strip())
    
    if N == 1:
        print(0)
        return
    
    sum_counts = 0
    L = 1
    
    while True:
        if L == 1:
            count = 10
        else:
            if L % 2 == 0:
                k = L // 2
                count = 9 * (10 ** (k - 1))
            else:
                k = (L - 1) // 2
                count = 9 * (10 ** k)
        
        if sum_counts < N <= sum_counts + count:
            pos_in_group = N - sum_counts - 1
            break
        else:
            sum_counts += count
            L += 1
    
    if L == 1:
        print(pos_in_group)
    else:
        if L % 2 == 0:
            k = L // 2
            base = 10 ** (k - 1) + pos_in_group
            s = str(base)
            palindrome = s + s[::-1]
        else:
            k = (L - 1) // 2
            base = 10 ** k + pos_in_group
            s = str(base)
            palindrome = s + s[:-1][::-1]
        print(palindrome)

find_nth_palindrome()