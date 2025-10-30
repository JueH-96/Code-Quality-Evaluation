def main():
    n = int(input().strip())
    if n == 1:
        print(0)
    elif n <= 10:
        print(n-1)
    else:
        n0 = n - 10
        k = 1
        while n0 > 99 * (10**(k-1)):
            n0 -= 99 * (10**(k-1))
            k += 1
        
        count_even = 9 * (10**(k-1))
        if n0 <= count_even:
            base = 10**(k-1)
            first_half = base + n0 - 1
            s = str(first_half)
            res = s + s[::-1]
            print(res)
        else:
            n0_odd = n0 - count_even
            base = 10**k
            first_half = base + n0_odd - 1
            s = str(first_half)
            res = s + s[:-1][::-1]
            print(res)

if __name__ == '__main__':
    main()