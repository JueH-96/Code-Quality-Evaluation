def main():
    import sys
    data = sys.stdin.readline().split()
    n = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    dishes = list(map(int, sys.stdin.readline().split()))
    min_dish = min(dishes)
    option_with_coupon = Q + min_dish
    answer = min(P, option_with_coupon)
    print(answer)

if __name__ == "__main__":
    main()