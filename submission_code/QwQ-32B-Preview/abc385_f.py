def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    buildings = []
    for i in range(N):
        X = int(data[1 + 2*i])
        H = int(data[2 + 2*i])
        buildings.append((X, H))
    
    buildings.sort(key=lambda x: x[0])
    
    h_min = 0.0
    for i in range(1, N):
        X_i, H_i = buildings[i]
        X_j, H_j = buildings[i-1]
        numerator = H_j * X_i - X_j * H_i
        denominator = X_i - X_j
        if denominator == 0:
            continue  # Skip if buildings are at the same X coordinate
        h_min_i = numerator / denominator
        if h_min_i < 0:
            h_min_i = 0.0
        h_min = max(h_min, h_min_i)
    
    if h_min > 0:
        print("{0:.10f}".format(h_min))
    else:
        print(-1)

if __name__ == "__main__":
    main()