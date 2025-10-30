import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	m = int(data[1])
	pieces = []
	R = set()
	C = set()
	D1 = set()
	D2 = set()
	
	index = 2
	for i in range(m):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		pieces.append((a, b))
		R.add(a)
		C.add(b)
		D1.add(a + b)
		D2.add(a - b)
	
	total_squares = n * n
	
	A = len(R) * n
	B = len(C) * n
	C_val = 0
	for s in D1:
		if 2 <= s <= 2 * n:
			if s <= n + 1:
				count_s = s - 1
			else:
				count_s = 2 * n + 1 - s
			C_val += count_s
	
	D_val = 0
	for d in D2:
		count_d = n - abs(d)
		D_val += count_d
	
	AB = len(R) * len(C)
	
	AC = 0
	for i in R:
		for s in D1:
			j = s - i
			if 1 <= j <= n:
				AC += 1
	
	AD = 0
	for i in R:
		for d in D2:
			j = i - d
			if 1 <= j <= n:
				AD += 1
	
	BC = 0
	for j in C:
		for s in D1:
			i_val = s - j
			if 1 <= i_val <= n:
				BC += 1
	
	BD = 0
	for j in C:
		for d in D2:
			i_val = j + d
			if 1 <= i_val <= n:
				BD += 1
	
	CD = 0
	for s in D1:
		for d in D2:
			if (s + d) % 2 == 0:
				i_val = (s + d) // 2
				j_val = (s - d) // 2
				if 1 <= i_val <= n and 1 <= j_val <= n:
					CD += 1
	
	ABC = 0
	for i in R:
		for j in C:
			s = i + j
			if s in D1:
				ABC += 1
	
	ABD = 0
	for i in R:
		for j in C:
			d = i - j
			if d in D2:
				ABD += 1
	
	ACD = 0
	for i in R:
		for s in D1:
			j = s - i
			if 1 <= j <= n:
				d = i - j
				if d in D2:
					ACD += 1
	
	BCD = 0
	for j in C:
		for s in D1:
			i_val = s - j
			if 1 <= i_val <= n:
				d = i_val - j
				if d in D2:
					BCD += 1
	
	ABCD = 0
	for i in R:
		for j in C:
			s = i + j
			d = i - j
			if s in D1 and d in D2:
				ABCD += 1
	
	attacked = (A + B + C_val + D_val) - (AB + AC + AD + BC + BD + CD) + (ABC + ABD + ACD + BCD) - ABCD
	safe = total_squares - attacked
	print(safe)

if __name__ == '__main__':
	main()