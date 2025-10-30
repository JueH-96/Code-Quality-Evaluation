import sys
data = sys.stdin.read().split()
N = int(data[0])
steps = list(map(int, data[1:]))
weekly_sums = [sum(steps[7*i : 7*i + 7]) for i in range(N)]
print(' '.join(map(str, weekly_sums)))