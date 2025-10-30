def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    heights = list(map(int, data[1:]))
    
    # The height of the first building
    first_height = heights[0]
    
    # Iterate through the buildings (starting from the second one)
    for i in range(1, N):
        if heights[i] > first_height:
            print(i + 1)  # Output 1-indexed position
            return
    print(-1)

if __name__ == '__main__':
    main()