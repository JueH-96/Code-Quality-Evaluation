import sys

def main():
    n = int(sys.stdin.readline())
    if n == 1:
        print(-1)
        return
    
    buildings = []
    for _ in range(n):
        x, h = map(int, sys.stdin.readline().split())
        buildings.append((x, h))
    
    hull = [buildings[0]]
    max_h = -float('inf')
    
    for i in range(1, n):
        xi, hi = buildings[i]
        current_max = -float('inf')
        # Compute current_max for current building i
        for (xj, hj) in hull:
            numerator = hj * xi - hi * xj
            denominator = xi - xj
            if abs(denominator) < 1e-10:
                continue  # avoid division by zero, though problem states X_i is strictly increasing
            h_ji = numerator / denominator
            if h_ji > current_max:
                current_max = h_ji
        
        if current_max > max_h:
            max_h = current_max
        
        # Maintain the convex hull by adding current building
        while len(hull) >= 2:
            o = hull[-2]
            a = hull[-1]
            # Calculate cross product of (a-o) and (current point -a)
            cross = (a[0] - o[0]) * (hi - a[1]) - (a[1] - o[1]) * (xi - a[0])
            if cross <= 1e-10:  # using epsilon to handle precision
                hull.pop()
            else:
                break
        hull.append((xi, hi))
    
    # Determine the result based on max_h
    epsilon = 1e-10
    if max_h > epsilon:
        print("{0:.15f}".format(max_h))
    elif max_h < -epsilon:
        print(-1)
    else:
        print("0.000000000000000")

if __name__ == "__main__":
    main()