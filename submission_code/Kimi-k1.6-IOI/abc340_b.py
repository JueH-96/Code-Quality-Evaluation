q = int(input())
A = []
output = []
for _ in range(q):
    parts = input().split()
    if parts[0] == '1':
        A.append(int(parts[1]))
    else:
        k = int(parts[1])
        output.append(A[-k])
print('
'.join(map(str, output)))