n = int(input().strip())
A = list(map(int, input().split()))
total = sum(A)
if n == 0:
    print("")
    exit(0)
max_val = max(A)

freq = [0] * (max_val + 1)

for num in A:
    freq[num] += 1

F = [0] * (max_val + 1)
for i in range(1, max_val + 1):
    F[i] = F[i-1] + i * freq[i]

res = [str(total - F[num]) for num in A]
print(" ".join(res))