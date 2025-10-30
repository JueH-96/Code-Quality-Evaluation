def main():
    import sys
    mod = 998244353
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Precompute powers for digit counts 1 to 11.
    # Since A[i] <= 10^9, the number of digits is at most 10.
    power_map = {}
    for d in range(1, 12):
        power_map[d] = pow(10, d, mod)

    result = 0
    prefix_sum = 0
    # We are summing for all pairs i<j:
    # f(A_i,A_j) = A_i * 10^(len(A_j)) + A_j.
    # For each j from 2 to N (1-indexed) we add:
    #   (10^(digits(A_j)) * sum_{i < j} A_i) + ((j-1) * A_j).
    # Process the list in order.
    for idx, val in enumerate(A):
        if idx > 0:
            # Compute the digit count (by string conversion, which is fast enough)
            d = len(str(val))
            # Contribution from the first term: For each previous element A_i, A_i * 10^(len(val))
            result = (result + power_map[d] * prefix_sum) % mod
            # Contribution from the second term: This current element is the second element in the pair for each previous index.
            result = (result + idx * val) % mod
        
        prefix_sum = (prefix_sum + val) % mod
    
    sys.stdout.write(str(result))


if __name__ == '__main__':
    main()