import sys
input = sys.stdin.read

def solve():
    data = input().split()
    t = int(data[0])
    index = 1

    results = []

    for _ in range(t):
        S = data[index]
        X = data[index + 1]
        Y = data[index + 2]
        index += 3

        if X.count('0') * Y.count('1') == X.count('1') * Y.count('0'):
            results.append("Yes")
        else:
            results.append("No")

    sys.stdout.write("
".join(results) + "
")

solve()