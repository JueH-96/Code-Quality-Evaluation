import sys
input = sys.stdin.read

def main():
    data = input().split()
    idx = 0
    
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    C = data[idx:idx+N]
    idx += N
    
    D = data[idx:idx+M]
    idx += M
    
    P = list(map(int, data[idx:idx+M+1]))
    
    price_map = {D[i]: P[i+1] for i in range(M)}
    default_price = P[0]
    
    total_price = 0
    for color in C:
        if color in price_map:
            total_price += price_map[color]
        else:
            total_price += default_price
    
    print(total_price)

if __name__ == "__main__":
    main()