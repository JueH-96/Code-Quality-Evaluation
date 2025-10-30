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
        index += 1
        P = list(map(int, data[index:index + N]))
        index += N

        # Find the maximum index where P[i] != i
        max_index = -1
        for i in range(N):
            if P[i] != i + 1:
                max_index = i

        # Find the minimum index where P[i] == max_index + 1
        min_index = -1
        for i in range(N):
            if P[i] == max_index + 1:
                min_index = i
                break

        # The minimum number of operations required
        if max_index == -1:
            results.append(0)
        else:
            results.append(min_index + 1)

    sys.stdout.write("
".join(map(str, results)) + "
")

solve()