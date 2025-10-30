# YOUR CODE HERE
def main():
    n = int(input())
    buildings = []
    for _ in range(n):
        x, h = map(int, input().split())
        buildings.append((x, h))
    
    threshold_heights = []
    
    for i in range(1, n):  # For each building (except the first)
        x_i, h_i = buildings[i]
        for j in range(i):  # Check if any previous building blocks its view
            x_j, h_j = buildings[j]
            
            # Calculate the threshold height for building j to block building i
            threshold_height = (h_j * x_i - h_i * x_j) / (x_i - x_j)
            threshold_heights.append(threshold_height)
    
    max_threshold_height = max(threshold_heights) if threshold_heights else -float('inf')
    
    if max_threshold_height < 0:
        return -1
    else:
        return max_threshold_height

print(main())