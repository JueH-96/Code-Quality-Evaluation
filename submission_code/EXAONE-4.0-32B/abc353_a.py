def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    heights = list(map(int, data[1].split()))
    
    if n <= 1:
        print(-1)
        return
        
    first_height = heights[0]
    for i in range(1, n):
        if heights[i] > first_height:
            print(i + 1)
            return
            
    print(-1)

if __name__ == "__main__":
    main()