def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    if not input_data:
        return
    N = int(input_data[0])
    points = []
    index = 1
    for i in range(N):
        x = int(input_data[index])
        y = int(input_data[index+1])
        index += 2
        points.append((x, y))
    
    # We'll output result for each point i
    # Using squared Euclidean distance to avoid float issues.
    results = []
    for i in range(N):
        xi, yi = points[i]
        max_sqdist = -1
        farthest_id = -1
        for j in range(N):
            if i == j:
                continue
            xj, yj = points[j]
            dx = xi - xj
            dy = yi - yj
            sqdist = dx*dx + dy*dy
            if sqdist > max_sqdist:
                max_sqdist = sqdist
                farthest_id = j + 1  # ID number is index+1
            elif sqdist == max_sqdist:
                # if tie, choose minimum ID:
                farthest_id = min(farthest_id, j + 1)
        results.append(farthest_id)
    
    # Print results line by line
    print("
".join(map(str, results)))

if __name__ == '__main__':
    main()