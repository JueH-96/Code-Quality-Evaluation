def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    MOD = 998244353

    # The number of ways to assign colors such that no two adjacent people (in a circle) share the same color
    # can be computed by the formula:
    #   (M-1)^N + (-1)^N*(M-1)
    # where the first term counts the sequences with distinct consecutive colors in a line and the second term
    # adjusts the count to meet the condition at the circular boundary.
    
    term = pow(M - 1, N, MOD)
    if N % 2 == 0:
        result = (term + (M - 1)) % MOD
    else:
        result = (term - (M - 1)) % MOD  # This might be negative, so we apply % MOD to adjust.
    
    sys.stdout.write(str(result))
    
if __name__ == '__main__':
    main()