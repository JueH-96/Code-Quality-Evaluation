def main():
    import sys
    input_data = sys.stdin.read().split()
    mod = 998244353

    n = int(input_data[0])
    A = list(map(int, input_data[1:]))

    # We need to compute the sum of f(A_i, A_j) for 1 <= i < j <= n.
    # Note that f(x, y) = int(str(x) + str(y)) = x * (10^(number of digits in y)) + y.
    # Thus the expression becomes:
    #   sum(i < j)[ A_i * 10^(digits(A_j)) + A_j ]
    # This equals: sum(j=2..n)[ (10^(digits(A_j)) * (sum of A_i for i < j)) + A_j*(j-1) ]
    # We can accumulate the prefix sum of A and use it for each A_j.
    
    prefix_sum = 0  # Sum of A[0] ... A[j-1]
    S1 = 0  # To store sum of A_i * 10^(digits(A_j))
    S2 = 0  # To store sum of A_j for each j with multiplicity (j-1)
    
    for index, number in enumerate(A):
        if index > 0:
            # Calculate digits of the current number (i.e., A_j)
            digit_count = len(str(number))
            # Compute 10^(digit_count) modulo mod
            power_val = pow(10, digit_count, mod)
            S1 = (S1 + prefix_sum * power_val) % mod
            S2 = (S2 + number * index) % mod  # index is (j - 1) because j = index+1
        
        prefix_sum = (prefix_sum + number) % mod
    
    result = (S1 + S2) % mod
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()