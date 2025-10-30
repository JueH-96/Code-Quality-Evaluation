def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    n = int(input_data[0])
    m = int(input_data[1])
    
    mod = 998244353

    # For a circle with n people and m colors (or integers),
    # the number of ways to assign integers such that no two adjacent persons
    # receive the same integer is given by:
    #
    #    (m-1)^n + (-1)^n * (m-1)
    #
    # where exponentiation is done under modulo arithmetic. We use Python's
    # pow function to perform fast exponentiation modulo 'mod'.

    term1 = pow(m - 1, n, mod)
    if n % 2 == 0:
        term2 = m - 1
    else:
        term2 = mod - (m - 1)  # effectively computes - (m-1) mod mod

    result = (term1 + term2) % mod
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()