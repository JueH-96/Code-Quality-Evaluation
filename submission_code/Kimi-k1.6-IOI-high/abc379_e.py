n = int(input())
s = input().strip()
digits = [int(c) for c in s]

sum_i = [0] * (n + 1)
for k in range(1, n + 1):
    sum_i[k] = sum_i[k - 1] + digits[k - 1] * k

result = []
carry = 0

for d in range(n):
    total = sum_i[n - d] + carry
    result.append(total % 10)
    carry = total // 10

while carry > 0:
    result.append(carry % 10)
    carry //= 10

# The result is currently in reverse order (least significant digit first)
result.reverse()
print(''.join(map(str, result)))