def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    heights = list(map(int, input().split()))
    
    first_height = heights[0]
    for idx in range(1, N):
        if heights[idx] > first_height:
            print(idx + 1)  # converting 0-index to 1-index
            return
    print(-1)

if __name__ == '__main__':
    main()