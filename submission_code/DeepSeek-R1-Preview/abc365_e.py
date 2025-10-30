n = int(input())
a = list(map(int, input().split()))

prefix_xor = [0] * (n + 1)
for i in range(1, n + 1):
    prefix_xor[i] = prefix_xor[i - 1] ^ a[i - 1]

cnt_bits = [0] * 31
for num in a:
    for k in range(31):
        if num & (1 << k):
            cnt_bits[k] += 1

result = 0
for k in range(31):
    count0 = 1  # Starting with prefix_xor[0] which is 0
    count1 = 0
    total = 0
    for b in range(1, n + 1):
        current_bit = (prefix_xor[b] >> k) & 1
        if current_bit == 0:
            total += count1
        else:
            total += count0
        if current_bit == 0:
            count0 += 1
        else:
            count1 += 1
    bad = cnt_bits[k]
    count = total - bad
    result += count * (1 << k)

print(result)