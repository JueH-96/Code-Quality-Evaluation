Q = int(input())
A = []
output_lines = []
for _ in range(Q):
    query = input().split()
    type = int(query[0])
    if type == 1:
        x = int(query[1])
        A.append(x)
    elif type == 2:
        k = int(query[1])
        index = len(A) - k
        output_lines.append(str(A[index]))

for line in output_lines:
    print(line)