# YOUR CODE HERE
def find_taller_building():
    import sys
    input = sys.stdin.read
    data = input().strip().split()
    
    N = int(data[0])
    heights = list(map(int, data[1:]))
    
    first_building_height = heights[0]
    
    for i in range(1, N):
        if heights[i] > first_building_height:
            print(i + 1)  # +1 to convert 0-based index to 1-based position
            return
    
    print(-1)

find_taller_building()