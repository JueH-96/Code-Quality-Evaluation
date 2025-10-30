n = int(input())
a = list(map(int, input().split()))
x = [0] * (n + 1)
for i in range(1, n + 1):
    x[i] = x[i-1] ^ a[i-1]

cnt = [0] * 31
# Initialize with x[0]
for b in range(31):
    if (x[0] >> b) & 1:
        cnt[b] += 1

total = 0
for j in range(2, n + 1):
    current_xor = x[j]
    contribution = 0
    for b in range(31):
        if (current_xor >> b) & 1:
            contribution += ( (j-1 - cnt[b]) ) * (1 << b)
        else:
            contribution += cnt[b] * (1 << b)
    total += contribution
    # Add x[j-1] to cnt
    for b in range(31):
        if (x[j-1] >> b) & 1:
            cnt[b] += 1

print(total)