# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])
S = data[2]
queries = [(int(data[3 + 2 * i]) - 1, int(data[4 + 2 * i])) for i in range(Q)]

result = []
for l, r in queries:
    count = 0
    for i in range(l, r - 1):
        if S[i] == S[i + 1]:
            count += 1
    result.append(count)

for ans in result:
    print(ans)