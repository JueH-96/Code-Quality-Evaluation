import sys
input = sys.stdin.read
data = input().split()

T = int(data[0])
index = 1
results = []

for _ in range(T):
    N = int(data[index])
    K = int(data[index + 1])
    index += 2

    if N % 2 == 0 and K % 2 == 0:
        results.append("Yes")
    else:
        results.append("No")

sys.stdout.write("
".join(results) + "
")