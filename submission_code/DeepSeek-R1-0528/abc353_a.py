def main():
    n = int(input().strip())
    heights = list(map(int, input().split()))
    
    first_height = heights[0]
    for i in range(1, n):
        if heights[i] > first_height:
            print(i + 1)
            return
    
    print(-1)

if __name__ == "__main__":
    main()