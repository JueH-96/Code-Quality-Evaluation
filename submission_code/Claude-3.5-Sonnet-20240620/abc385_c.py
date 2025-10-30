# YOUR CODE HERE
def max_decorated_buildings(N, heights):
    max_count = 1
    
    for i in range(N):
        for j in range(i+1, N):
            if heights[i] == heights[j]:
                interval = j - i
                count = 2
                next_pos = j + interval
                
                while next_pos < N and heights[next_pos] == heights[i]:
                    count += 1
                    next_pos += interval
                
                max_count = max(max_count, count)
    
    return max_count

# Read input
N = int(input())
heights = list(map(int, input().split()))

# Solve and print output
print(max_decorated_buildings(N, heights))