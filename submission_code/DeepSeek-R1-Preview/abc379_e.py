N = int(input())
S = input().strip()

contrib = [0] * N
for k in range(N):
    digit = int(S[k])
    contrib[k] = digit * (k + 1)

# Compute prefix sums
prefix_sum = [0] * N
prefix_sum[0] = contrib[0]
for k in range(1, N):
    prefix_sum[k] = prefix_sum[k-1] + contrib[k]

digits = []

for p in range(N):
    idx = N - 1 - p
    sum_p = prefix_sum[idx] if idx >= 0 else 0
    if sum_p == 0:
        continue
    s = str(sum_p)
    for i, c in enumerate(reversed(s)):
        digit = int(c)
        pos = p + i
        if pos >= len(digits):
            digits.extend([0] * (pos - len(digits) + 1))
        digits[pos] += digit
        carry = digits[pos] // 10
        digits[pos] %= 10
        while carry > 0:
            pos += 1
            if pos >= len(digits):
                digits.append(carry)
                carry = 0
            else:
                digits[pos] += carry
                carry = digits[pos] // 10
                digits[pos] %= 10

# Convert digits to the final result string
if not digits:
    print(0)
else:
    # Reverse to get the correct order and strip leading zeros
    digits.reverse()
    result = ''.join(map(str, digits)).lstrip('0')
    print(result if result else '0')