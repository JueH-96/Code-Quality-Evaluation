# YOUR CODE HERE
def max_decorated_buildings(N, heights):
    max_count = 1
    
    for i in range(N):
        for j in range(i + 1, N):
            if heights[j] == heights[i]:
                interval = j - i
                count = 2
                k = j + interval
                while k < N:
                    if heights[k] == heights[i]:
                        count += 1
                    k += interval
                max_count = max(max_count, count)
    
    return max_count

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    heights = list(map(int, data[1:]))
    print(max_decorated_buildings(N, heights))