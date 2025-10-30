def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    idx = 0
    N = int(data[idx])
    M = int(data[idx+1])
    idx +=2
    
    C = data[idx:idx+N]
    idx +=N
    
    D = data[idx:idx+M]
    idx +=M
    
    P = list(map(int, data[idx:idx+M+1]))
    P0 = P[0]
    price_map = {color: price for color, price in zip(D, P[1:])}
    
    total = 0
    for color in C:
        total += price_map.get(color, P0)
    
    print(total)

if __name__ == "__main__":
    main()