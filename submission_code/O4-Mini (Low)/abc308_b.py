def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    C = data[2:2+N]
    D = data[2+N:2+N+M]
    P_vals = list(map(int, data[2+N+M:2+N+M+M+1]))
    P0 = P_vals[0]
    # Build price map for known colors
    price_map = {D[i]: P_vals[i+1] for i in range(M)}
    total = 0
    for color in C:
        total += price_map.get(color, P0)
    print(total)

if __name__ == "__main__":
    main()