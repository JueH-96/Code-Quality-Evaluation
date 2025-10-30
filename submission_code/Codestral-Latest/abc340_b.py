import sys
input = sys.stdin.read
data = input().split()

Q = int(data[0])
A = []
output = []

i = 1
while i <= Q:
    query_type = int(data[i])
    if query_type == 1:
        x = int(data[i + 1])
        A.append(x)
        i += 2
    elif query_type == 2:
        k = int(data[i + 1])
        output.append(A[-k])
        i += 2

for result in output:
    print(result)