n = int(input())
if n == 1:
    print(0)
else:
    n_prime = n - 1
    if n_prime <= 9:
        print(n_prime)
    else:
        n_double_prime = n_prime - 9
        k = 1
        while True:
            even_count = 9 * (10 ** (k - 1))
            if n_double_prime <= even_count:
                m = n_double_prime
                root = 10 ** (k - 1) + m - 1
                reverse_root = int(str(root)[::-1])
                palindrome = root * (10 ** k) + reverse_root
                print(palindrome)
                break
            else:
                n_double_prime -= even_count
            odd_count = 9 * (10 ** k)
            if n_double_prime <= odd_count:
                m = n_double_prime
                root = 10 ** k + m - 1
                palindrome_str = str(root) + str(root)[:-1][::-1]
                palindrome = int(palindrome_str)
                print(palindrome)
                break
            else:
                n_double_prime -= odd_count
            k += 1