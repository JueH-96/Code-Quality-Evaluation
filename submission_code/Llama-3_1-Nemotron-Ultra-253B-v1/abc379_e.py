n = int(input())
s = input().strip()

digits = [0] * (n + 7)  # To prevent index out of range

for i in range(n):
    d = int(s[i])
    val = (i + 1) * d
    k = n - i
    j = 0
    while val > 0:
        digits[k + j] += val % 10
        val = val // 10
        j += 1

# Process carries
carry = 0
for i in range(len(digits) - 1, -1, -1):
    carry += digits[i]
    digits[i] = carry % 10
    carry = carry // 10

if carry > 0:
    digits = [carry] + digits

# Convert digits to sum_geo_part
sum_geo_part = 0
highest = len(digits) - 1
while highest >= 0 and digits[highest] == 0:
    highest -= 1
if highest < 0:
    sum_geo_part = 0
else:
    for i in range(highest, -1, -1):
        sum_geo_part = sum_geo_part * 10 + digits[i]

sum_minus = 0
for i in range(n):
    sum_minus += (i + 1) * int(s[i])

ans = (sum_geo_part - sum_minus) // 9
print(ans)