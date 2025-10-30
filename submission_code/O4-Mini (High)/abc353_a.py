def main():
    N = int(input().strip())
    heights = list(map(int, input().split()))
    first = heights[0]
    for i, h in enumerate(heights[1:], start=2):
        if h > first:
            print(i)
            return
    print(-1)

main()