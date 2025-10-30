def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    buildings = []
    index = 1
    for _ in range(N):
        X = int(data[index])
        H = int(data[index + 1])
        buildings.append((X, H))
        index += 2
    
    buildings.sort()
    
    max_h = 0.0
    for i in range(N):
        X_i, H_i = buildings[i]
        max_h_i = -float('inf')
        for j in range(i):
            X_j, H_j = buildings[j]
            numerator = H_j * X_i - H_i * X_j
            denominator = X_i - X_j
            if denominator == 0:
                continue
            h_ji = numerator / denominator
            if h_ji > max_h_i:
                max_h_i = h_ji
        if max_h_i > max_h:
            max_h = max_h_i
    
    if max_h > 0:
        print("{0:.18f}".format(max_h))
    else:
        # Check if at h=0, all buildings are visible
        visible = True
        for i in range(N):
            X_i, H_i = buildings[i]
            # Check if there exists a Q on building i such that PQ doesn't intersect any other building
            # Choose Q at (X_i, H_i)
            # The line from P(0,0) to Q(X_i, H_i) is y = (H_i / X_i) * x
            # Check if this line intersects any building j with X_j < X_i at y >= H_j
            blocked = False
            for j in range(i):
                X_j, H_j = buildings[j]
                y_j = (H_i / X_i) * X_j
                if y_j >= H_j:
                    blocked = True
                    break
            if not blocked:
                continue
            else:
                visible = False
                break
        if visible:
            print("-1")
        else:
            print("0.000000000000000000")

if __name__ == '__main__':
    main()