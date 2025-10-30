# YOUR CODE HERE
N = int(input())
H = list(map(int, input().split()))
first_building_height = H[0]
for i in range(1, N):
    if H[i] > first_building_height:
        print(i+1)
        break
else:
    print(-1)