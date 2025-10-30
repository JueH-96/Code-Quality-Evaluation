def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D = list(map(int, data[3:]))

    # If using the coupon, total cost is Q + the cheapest dish
    # Otherwise, just pay the regular price P
    ans = min(P, Q + min(D))
    print(ans)

# DON'T forget to call main function
if __name__ == '__main__':
    main()