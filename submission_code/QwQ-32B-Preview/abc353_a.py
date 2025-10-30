N = int(input())
heights = list(map(int, input().split()))
first_height = heights[0]
for i in range(1, N):
    if heights[i] > first_height:
        print(i + 1)
        break
else:
    print(-1)