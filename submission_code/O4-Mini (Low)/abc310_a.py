def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:]))

    # Cost without coupon
    best = P
    # Cost with coupon + one dish
    best = min(best, Q + min(D))
    print(best)

if __name__ == "__main__":
    main()