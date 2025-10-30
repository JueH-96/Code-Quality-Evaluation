def main():
    import sys
    data = sys.stdin.read().split()
    # Parse inputs
    N = int(data[0])
    M = int(data[1])
    idx = 2
    C = data[idx:idx+N]
    idx += N
    D = data[idx:idx+M]
    idx += M
    P = list(map(int, data[idx:idx+M+1]))
    # Default price
    P0 = P[0]
    # Build price map for colors
    price_map = {D[i]: P[i+1] for i in range(M)}
    # Sum total
    total = 0
    for color in C:
        total += price_map.get(color, P0)
    # Output result
    print(total)

if __name__ == "__main__":
    main()