import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

index = 2
rows = {}
cols = {}

for _ in range(M):
    x = int(data[index])
    y = int(data[index + 1])
    c = data[index + 2]
    index += 3

    if c == 'B':
        if x in rows:
            rows[x].append(y)
        else:
            rows[x] = [y]

        if y in cols:
            cols[y].append(x)
        else:
            cols[y] = [x]

def check_valid(positions):
    positions.sort()
    for i in range(len(positions) - 1):
        if positions[i] + 1 != positions[i + 1]:
            return False
    return True

for row in rows.values():
    if not check_valid(row):
        print("No")
        exit()

for col in cols.values():
    if not check_valid(col):
        print("No")
        exit()

print("Yes")