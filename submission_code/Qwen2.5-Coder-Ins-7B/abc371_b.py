# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

families = [0] * (N + 1)

for i in range(M):
    family = int(data[2 + 2 * i])
    gender = data[3 + 2 * i]
    if gender == 'M':
        if families[family] == 0:
            families[family] = 1
            print("Yes")
        else:
            print("No")
    else:
        print("No")