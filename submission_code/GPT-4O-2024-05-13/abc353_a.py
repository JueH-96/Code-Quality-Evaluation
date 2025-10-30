# YOUR CODE HERE
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
H = list(map(int, data[1:]))

first_height = H[0]

for i in range(1, N):
    if H[i] > first_height:
        print(i + 1)
        break
else:
    print(-1)