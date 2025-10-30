# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    points = []
    for i in range(N):
        x = int(data[1 + 2*i])
        y = int(data[2 + 2*i])
        points.append((x, y))
    
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
        # Calculate the sum of |u_i - u_j| / 2 for all pairs
        # Since u_i and u_j are even or odd, |u_i - u_j| is even
        # So |u_i - u_j| / 2 is integer
        # To compute the sum efficiently, we can use the prefix sum approach
        u_list = [u for u, v in transformed]
        # Sort u_list
        u_list.sort()
        # Compute prefix sums
        prefix = [0] * (len(u_list) + 1)
        for i in range(len(u_list)):
            prefix[i+1] = prefix[i] + u_list[i]
        # Calculate the sum of |u_i - u_j| for all i < j
        total = 0
        for i in range(len(u_list)):
            total += u_list[i] * i - prefix[i]
        # Since |u_i - u_j| / 2 is the distance, we divide the total by 2
        return total // 2
    
    # Calculate the sum for both groups
    sum0 = calculate_distance(group0)
    sum1 = calculate_distance(group1)
    
    # The total sum is the sum of distances within group0 and group1
    total_sum = sum0 + sum1
    
    print(total_sum)

if __name__ == "__main__":
    main()