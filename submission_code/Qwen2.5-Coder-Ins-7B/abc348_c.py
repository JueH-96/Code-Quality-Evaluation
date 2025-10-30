# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
beans = []

for i in range(N):
    A = int(data[2*i+1])
    C = int(data[2*i+2])
    beans.append((A, C))

beans.sort()

max_min_deliciousness = 0

for i in range(N):
    min_deliciousness = beans[i][0]
    for j in range(i, N):
        if beans[j][1] != beans[i][1]:
            break
        min_deliciousness = max(min_deliciousness, beans[j][0])
    max_min_deliciousness = max(max_min_deliciousness, min_deliciousness)

print(max_min_deliciousness)