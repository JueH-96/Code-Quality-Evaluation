n = int(input().strip())
s = input().strip()
A = [1 if c == '1' else 0 for c in s]

total = 0
T_prev = 0
for j in range(n):
	T_current = A[j] + j - A[j] * T_prev
	total += T_current
	T_prev = T_current

print(total)