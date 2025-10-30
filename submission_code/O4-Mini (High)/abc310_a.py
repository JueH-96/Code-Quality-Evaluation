def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:3+N]))
    # Without coupon: pay P.
    # With coupon: pay Q + min(D).
    ans = min(P, Q + min(D))
    print(ans)

if __name__ == "__main__":
    main()