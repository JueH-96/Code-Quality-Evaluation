import sys
import bisect

def count_valid_points(N, D, points):
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    
    x_coords.sort()
    y_coords.sort()
    
    # Precompute prefix sums for x and y coordinates
    prefix_x = [0] * (N + 1)
    prefix_y = [0] * (N + 1)
    
    for i in range(N):
        prefix_x[i+1] = prefix_x[i] + x_coords[i]
        prefix_y[i+1] = prefix_y[i] + y_coords[i]
    
    # Function to calculate the sum of absolute differences for a given x or y
    def sum_abs_diff(sorted_coords, prefix, val):
        idx = bisect.bisect_left(sorted_coords, val)
        sum_left = val * idx - prefix[idx]
        sum_right = (prefix[N] - prefix[idx]) - val * (N - idx)
        return sum_left + sum_right
    
    # Determine the range of x and y to consider
    # Since the sum of absolute differences is minimized at the median, we can limit the search around the median
    median_x = x_coords[N//2]
    median_y = y_coords[N//2]
    
    # We need to find all x and y such that sum_abs_diff(x) + sum_abs_diff(y) <= D
    # To find the range of x and y, we can iterate over possible x and for each x, find the range of y that satisfies the condition
    
    # First, find the range of x
    # The sum_abs_diff(x) is minimized at the median_x, and increases as we move away
    # So, we can find the x values where sum_abs_diff(x) <= D - sum_abs_diff(y_min)
    # But since y_min is not known, we can iterate x in a reasonable range around the median
    
    # Let's find the maximum possible sum_abs_diff(x) that allows at least one y to satisfy the condition
    # The minimum sum_abs_diff(y) is 0 (when y is the median_y)
    # So, sum_abs_diff(x) <= D
    
    # So, we can iterate x in the range [median_x - D, median_x + D]
    
    # Similarly for y, for each x, we can find the y such that sum_abs_diff(y) <= D - sum_abs_diff(x)
    
    # Now, implement this logic
    
    # First, find all x in [median_x - D, median_x + D] where sum_abs_diff(x) <= D
    x_min = median_x - D
    x_max = median_x + D
    
    valid_x = []
    for x in range(x_min, x_max + 1):
        sum_x = sum_abs_diff(x_coords, prefix_x, x)
        if sum_x <= D:
            valid_x.append((x, sum_x))
    
    # Now, for each valid x, find the range of y such that sum_abs_diff(y) <= D - sum_x
    total = 0
    for x, sum_x in valid_x:
        remaining_D = D - sum_x
        if remaining_D < 0:
            continue
        # Find y such that sum_abs_diff(y) <= remaining_D
        # The sum_abs_diff(y) is minimized at median_y, and increases as we move away
        # So, we can find the y in [median_y - remaining_D, median_y + remaining_D]
        y_min = median_y - remaining_D
        y_max = median_y + remaining_D
        
        # Now, find all y in [y_min, y_max] where sum_abs_diff(y) <= remaining_D
        # Since sum_abs_diff(y) is minimized at median_y, and increases as we move away, we can find the y in [y_min, y_max] where sum_abs_diff(y) <= remaining_D
        # To find the exact range, we can use binary search
        
        # Find the leftmost y where sum_abs_diff(y) <= remaining_D
        left = y_min
        right = median_y
        while left < right:
            mid = (left + right) // 2
            sum_y = sum_abs_diff(y_coords, prefix_y, mid)
            if sum_y <= remaining_D:
                right = mid
            else:
                left = mid + 1
        y_low = left
        
        # Find the rightmost y where sum_abs_diff(y) <= remaining_D
        left = median_y
        right = y_max
        while left < right:
            mid = (left + right + 1) // 2
            sum_y = sum_abs_diff(y_coords, prefix_y, mid)
            if sum_y <= remaining_D:
                left = mid
            else:
                right = mid - 1
        y_high = left
        
        # Now, count the number of y in [y_low, y_high]
        if y_low > y_high:
            continue
        total += y_high - y_low + 1
    
    return total

def main():
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    points = []
    index = 2
    for _ in range(N):
        x = int(data[index])
        y = int(data[index+1])
        points.append((x, y))
        index += 2
    print(count_valid_points(N, D, points))

if __name__ == "__main__":
    main()