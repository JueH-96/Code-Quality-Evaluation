def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    C = data[2:2+N]
    D = data[2+N:2+N+M]
    P = list(map(int, data[2+N+M:]))

    # P[0] corresponds to P_0, then P[1..M] correspond to P_1..P_M
    price_map = {}
    for i in range(M):
        price_map[D[i]] = P[i+1]
        
    default_price = P[0]
    
    total = 0
    for plate_color in C:
        if plate_color in price_map:
            total += price_map[plate_color]
        else:
            total += default_price
    
    print(total)

# Do not forget to call main function
if __name__ == "__main__":
    main()