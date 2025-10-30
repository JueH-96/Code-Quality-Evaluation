n = int(input().strip())
S = input().strip()
C = list(map(int, input().split()))

prefix0 = [0] * n
prefix1 = [0] * n

if S[0] == '0':
	prefix0[0] = 0
else:
	prefix0[0] = C[0]

if S[0] == '1':
	prefix1[0] = 0
else:
	prefix1[0] = C[0]

for i in range(1, n):
	if i % 2 == 0:
		bit0 = '0'
		bit1 = '1'
	else:
		bit0 = '1'
		bit1 = '0'
	add0 = C[i] if S[i] != bit0 else 0
	prefix0[i] = prefix0[i-1] + add0
	add1 = C[i] if S[i] != bit1 else 0
	prefix1[i] = prefix1[i-1] + add1

suf0 = [0] * (n + 1)
suf1 = [0] * (n + 1)
suf0[n] = 0
suf1[n] = 0

for i in range(n-1, -1, -1):
	if i % 2 == 0:
		bit0 = '0'
		bit1 = '1'
	else:
		bit0 = '1'
		bit1 = '0'
	add0 = C[i] if S[i] != bit0 else 0
	add1 = C[i] if S[i] != bit1 else 0
	suf0[i] = add0 + suf0[i+1]
	suf1[i] = add1 + suf1[i+1]

min_total = float('inf')

for i in range(0, n-1):
	for b in ['0', '1']:
		cost_mid = 0
		if S[i] != b:
			cost_mid += C[i]
		if S[i+1] != b:
			cost_mid += C[i+1]
			
		cost_prefix = 0
		if i > 0:
			desired_bit = '1' if b == '0' else '0'
			if (i-1) % 2 == 0:
				bit0_val = '0'
			else:
				bit0_val = '1'
			if bit0_val == desired_bit:
				cost_prefix = prefix0[i-1]
			else:
				cost_prefix = prefix1[i-1]
				
		cost_suffix = 0
		if i+2 < n:
			desired_bit = '1' if b == '0' else '0'
			if (i+2) % 2 == 0:
				bit0_val = '0'
			else:
				bit0_val = '1'
			if bit0_val == desired_bit:
				cost_suffix = suf0[i+2]
			else:
				cost_suffix = suf1[i+2]
				
		total_cost = cost_prefix + cost_mid + cost_suffix
		if total_cost < min_total:
			min_total = total_cost

print(min_total)