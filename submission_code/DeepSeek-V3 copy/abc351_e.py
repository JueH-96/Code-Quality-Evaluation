def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = []
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index+1])
        points.append((x, y))
        index += 2
    
    # Separate points into two groups based on (x + y) parity
    group0 = []
    group1 = []
    for x, y in points:
        if (x + y) % 2 == 0:
            group0.append((x, y))
        else:
            group1.append((x, y))
    
    # Function to calculate the sum of distances within a group
    def calculate_distance(group):
        if not group:
            return 0
        # Transform coordinates to (u, v) where u = x + y, v = x - y
        transformed = []
        for x, y in group:
            u = x + y
            v = x - y
            transformed.append((u, v))
        # Sort by u
        transformed.sort()
        # Calculate the sum of |u_i - u_j| + |v_i - v_j| for all pairs
        # Since u and v are independent, we can calculate separately
        u_list = [u for u, v in transformed]
        v_list = [v for u, v in transformed]
        # Calculate sum of |u_i - u_j| for all pairs
        u_list.sort()
        n = len(u_list)
        prefix_u = [0] * n
        prefix_u[0] = u_list[0]
        for i in range(1, n):
            prefix_u[i] = prefix_u[i-1] + u_list[i]
        sum_u = 0
        for i in range(n):
            sum_u += u_list[i] * i - prefix_u[i-1] if i > 0 else 0
        # Calculate sum of |v_i - v_j| for all pairs
        v_list.sort()
        prefix_v = [0] * n
        prefix_v[0] = v_list[0]
        for i in range(1, n):
            prefix_v[i] = prefix_v[i-1] + v_list[i]
        sum_v = 0
        for i in range(n):
            sum_v += v_list[i] * i - prefix_v[i-1] if i > 0 else 0
        # Total distance is (sum_u + sum_v) // 2
        return (sum_u + sum_v) // 2
    
    # Calculate distances within each group
    distance0 = calculate_distance(group0)
    distance1 = calculate_distance(group1)
    
    # Total distance is the sum of distances within each group
    total_distance = distance0 + distance1
    
    print(total_distance)

if __name__ == "__main__":
    main()