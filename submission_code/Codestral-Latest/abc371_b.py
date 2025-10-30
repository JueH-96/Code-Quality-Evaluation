# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

eldest_son = [-1] * (N + 1)

for i in range(M):
    family = int(data[2 * i + 2])
    gender = data[2 * i + 3]

    if gender == 'M':
        if eldest_son[family] == -1:
            eldest_son[family] = i

for i in range(M):
    family = int(data[2 * i + 2])
    gender = data[2 * i + 3]

    if gender == 'M' and eldest_son[family] == i:
        print("Yes")
    else:
        print("No")