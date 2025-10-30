import sys
input = sys.stdin.read

def solve():
    data = input().split()
    index = 0
    T = int(data[index])
    index += 1
    results = []

    for _ in range(T):
        N = int(data[index])
        M = int(data[index + 1])
        index += 2
        boxes = []

        for i in range(N):
            V = int(data[index])
            P = int(data[index + 1])
            index += 2
            boxes.append((V, P))

        boxes.sort(key=lambda x: (x[1], -x[0]))

        max_money = 0
        current_cost = 0
        current_capacity = 0

        for V, P in boxes:
            if current_capacity + V > M:
                break
            current_capacity += V
            current_cost += P
            max_money = max(max_money, current_capacity - current_cost)

        results.append(max_money)

    sys.stdout.write("
".join(map(str, results)) + "
")

solve()