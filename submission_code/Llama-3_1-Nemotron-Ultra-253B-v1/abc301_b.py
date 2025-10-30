n = int(input())
a = list(map(int, input().split()))
current = a.copy()

while True:
    found = False
    for i in range(len(current) - 1):
        x = current[i]
        y = current[i+1]
        if abs(x - y) != 1:
            if x < y:
                insert = list(range(x + 1, y))
            else:
                insert = list(range(x - 1, y, -1))
            current = current[:i+1] + insert + current[i+1:]
            found = True
            break
    if not found:
        break

print(' '.join(map(str, current)))