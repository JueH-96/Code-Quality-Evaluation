def main():
    import sys
    N = int(sys.stdin.readline())
    # Let k = floor(log2(N)), M = 2^k, r = N - M.
    # One can show the total cost is N*k + 2*r.
    k = N.bit_length() - 1
    M = 1 << k
    r = N - M
    ans = N * k + 2 * r
    print(ans)

if __name__ == "__main__":
    main()