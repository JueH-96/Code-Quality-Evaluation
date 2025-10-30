A = []
output = []
Q = int(input())
for _ in range(Q):
    parts = input().split()
    if parts[0] == '1':
        x = int(parts[1])
        A.append(x)
    else:
        k = int(parts[1])
        output.append(str(A[-k]))
print('
'.join(output))