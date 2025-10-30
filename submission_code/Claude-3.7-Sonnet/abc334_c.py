def find_min_weirdness(n, k, lost_colors):
    # Initialize array with 2 socks for each color
    socks_count = [2] * (n + 1)
    
    # Update counts based on lost socks
    for color in lost_colors:
        socks_count[color] -= 1
    
    # Create sorted list of all socks
    sock_list = []
    for color in range(1, n + 1):
        for _ in range(socks_count[color]):
            sock_list.append(color)
    
    # If even number of socks, pair adjacent socks
    if len(sock_list) % 2 == 0:
        total_weirdness = 0
        for i in range(0, len(sock_list), 2):
            total_weirdness += abs(sock_list[i] - sock_list[i + 1])
        return total_weirdness
    
    # If odd number of socks, try leaving each sock unpaired
    min_weirdness = float('inf')
    
    for i in range(len(sock_list)):
        # Create a new list without the sock at position i
        remaining = sock_list[:i] + sock_list[i+1:]
        
        # Calculate weirdness of pairs from remaining socks
        weirdness = 0
        for j in range(0, len(remaining), 2):
            weirdness += abs(remaining[j] - remaining[j + 1])
        
        min_weirdness = min(min_weirdness, weirdness)
    
    return min_weirdness

# Read input
n, k = map(int, input().split())
lost_colors = list(map(int, input().split()))

# Print output
print(find_min_weirdness(n, k, lost_colors))