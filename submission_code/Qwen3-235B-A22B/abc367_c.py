n, k = map(int, input().split())
r = list(map(int, input().split()))

results = []

def backtrack(pos, current, total):
    if pos == n:
        if total % k == 0:
            results.append(current.copy())
        return
    for i in range(1, r[pos] + 1):
        current.append(i)
        backtrack(pos + 1, current, total + i)
        current.pop()

backtrack(0, [], 0)

for seq in results:
    print(' '.join(map(str, seq)))