import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:]))

    # Create a dictionary to quickly find the next firework day
    firework_days = set(A)

    result = []
    for i in range(1, N + 1):
        days_until_fireworks = 0
        while i + days_until_fireworks not in firework_days:
            days_until_fireworks += 1
        result.append(days_until_fireworks)

    for r in result:
        print(r)

solve()