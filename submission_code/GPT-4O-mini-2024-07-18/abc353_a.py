# YOUR CODE HERE
def find_leftmost_taller_building():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    heights = list(map(int, data[1:N+1]))
    
    first_building_height = heights[0]
    
    for i in range(1, N):
        if heights[i] > first_building_height:
            print(i + 1)  # +1 for 1-based index
            return
    
    print(-1)

find_leftmost_taller_building()