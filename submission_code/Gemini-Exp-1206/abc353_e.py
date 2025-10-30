def f(x, y):
    count = 0
    for i in range(min(len(x), len(y))):
        if x[i] == y[i]:
            count += 1
        else:
            break
    return count

N = int(input())
S = input().split()
total = 0
for i in range(N - 1):
    for j in range(i + 1, N):
        total += f(S[i], S[j])
print(total)