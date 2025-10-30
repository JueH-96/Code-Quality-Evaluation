def main():
    data = input().split()
    n = int(data[0])
    P = int(data[1])
    Q = int(data[2])
    D_list = list(map(int, input().split()))
    
    min_dish = min(D_list)
    total_with_coupon = Q + min_dish
    answer = min(P, total_with_coupon)
    print(answer)

if __name__ == '__main__':
    main()