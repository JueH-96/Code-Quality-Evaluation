import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1

for _ in range(T):
    N = int(data[index])
    M = int(data[index + 1])
    index += 2
    boxes = []

    for i in range(N):
        V = int(data[index])
        P = int(data[index + 1])
        boxes.append((V, P))
        index += 2

    boxes.sort(key=lambda x: x[1] / x[0])

    money = 0
    for V, P in boxes:
        if P <= V:
            money += P
        else:
            break

    print(money)