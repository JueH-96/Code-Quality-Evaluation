def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    heights = list(map(int, data[1:]))
    
    first_height = heights[0]
    
    for i in range(1, N):
        if heights[i] > first_height:
            print(i + 1)
            return
    
    print(-1)

if __name__ == "__main__":
    main()