# YOUR CODE HERE
N = int(input())
H = list(map(int, input().split()))
first_height = H[0]
position = -1
for i in range(1, N):
    if H[i] > first_height:
        position = i + 1
        break
print(position)