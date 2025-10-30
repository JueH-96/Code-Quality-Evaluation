n = int(input().strip())
s = input().strip()
total = 0
F = 0
for i in range(n):
	digit = int(s[i])
	F = 10 * F + (i + 1) * digit
	total += F
print(total)