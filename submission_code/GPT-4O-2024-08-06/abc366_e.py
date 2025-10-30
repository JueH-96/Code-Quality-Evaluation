# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    
    points = []
    index = 2
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        points.append((x, y))
        index += 2
    
    # Transform points to (u, v) space
    u_values = [x + y for x, y in points]
    v_values = [x - y for x, y in points]
    
    # Calculate the range of u and v
    min_u = min(u_values)
    max_u = max(u_values)
    min_v = min(v_values)
    max_v = max(v_values)
    
    # Calculate the possible range for u and v
    u_range_min = min_u - D
    u_range_max = max_u + D
    v_range_min = min_v - D
    v_range_max = max_v + D
    
    # Count valid (u, v) pairs
    count = 0
    for u in range(u_range_min, u_range_max + 1):
        for v in range(v_range_min, v_range_max + 1):
            # Calculate the sum of distances in (u, v) space
            total_distance = sum(abs(u - u_i) + abs(v - v_i) for u_i, v_i in zip(u_values, v_values))
            if total_distance <= D:
                count += 1
    
    print(count)

if __name__ == "__main__":
    main()