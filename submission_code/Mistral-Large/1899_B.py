import sys
input = sys.stdin.read
data = input().split()

t = int(data[0])
index = 1
results = []

for _ in range(t):
    n = int(data[index])
    index += 1
    a = list(map(int, data[index:index + n]))
    index += n

    max_diff = 0

    for k in range(1, n + 1):
        if n % k == 0:
            trucks = [sum(a[i:i + k]) for i in range(0, n, k)]
            max_diff = max(max_diff, max(trucks) - min(trucks))

    results.append(max_diff)

sys.stdout.write("
".join(map(str, results)) + "
")