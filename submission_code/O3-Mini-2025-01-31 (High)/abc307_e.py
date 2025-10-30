def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    mod = 998244353

    # In a circle, the number of valid assignments (proper colorings) is given by:
    # (M - 1)^N + (-1)^N * (M - 1)
    # Explanation:
    # For a cycle graph with N vertices and M colors, one can show that the number of ways
    # to assign colors (from 0 to M-1) to the vertices such that no two adjacent vertices have 
    # the same color is (M-1)^N + (-1)^N*(M-1).
    
    M_minus_1 = M - 1
    result = pow(M_minus_1, N, mod)
    if N % 2 == 0:
        result = (result + M_minus_1) % mod
    else:
        result = (result - M_minus_1) % mod
    
    sys.stdout.write(str(result))

if __name__ == '__main__':
    main()