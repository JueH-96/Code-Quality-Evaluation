def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    buildings = []
    index = 1
    for i in range(N):
        X = int(data[index])
        h = int(data[index + 1])
        buildings.append((X, h))
        index += 2
    
    buildings.sort()
    max_m = 0.0
    for i in range(len(buildings)):
        X_i, h_i = buildings[i]
        current_max = 0.0
        for j in range(i):
            X_j, h_j = buildings[j]
            numerator = h_j * X_i - h_i * X_j
            denominator = X_i - X_j
            if denominator == 0:
                continue
            m = numerator / denominator
            if m > current_max:
                current_max = m
        if current_max > max_m:
            max_m = current_max
    print("{0:.15f}".format(max_m))

if __name__ == "__main__":
    main()