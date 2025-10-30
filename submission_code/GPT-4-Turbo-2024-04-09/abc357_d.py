def main():
    import sys
    input = sys.stdin.read
    MOD = 998244353
    
    N = int(input().strip())
    # Convert N to string
    N_str = str(N)
    # Repeat the string N times
    VN_str = N_str * N
    # Convert back to integer
    VN = int(VN_str)
    # Output the result modulo 998244353
    print(VN % MOD)

if __name__ == "__main__":
    main()