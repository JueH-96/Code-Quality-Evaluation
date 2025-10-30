# YOUR CODE HERE
Q = int(input())
A = []
output = []
for _ in range(Q):
    query = input().split()
    if query[0] == '1':
        x = int(query[1])
        A.append(x)
    else:
        k = int(query[1])
        output.append(str(A[-k]))
print('
'.join(output))