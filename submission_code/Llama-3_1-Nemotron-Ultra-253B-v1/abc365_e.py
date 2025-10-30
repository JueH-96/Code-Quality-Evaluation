n = int(input())
a = list(map(int, input().split()))

prefix_xor = [0] * (n + 1)
for i in range(1, n + 1):
    prefix_xor[i] = prefix_xor[i-1] ^ a[i-1]

sum1 = 0
for k in range(31):
    cnt0 = 1  # Initially, prefix_xor[0] has all bits 0
    cnt1 = 0
    total = 0
    for j in range(1, n + 1):
        current = prefix_xor[j]
        bit = (current >> k) & 1
        if bit:
            total += cnt0
        else:
            total += cnt1
        # Update counts for this bit
        if bit:
            cnt1 += 1
        else:
            cnt0 += 1
    sum1 += total * (1 << k)

sum2 = sum(a)
print(sum1 - sum2)