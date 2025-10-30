import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	m = int(data[1])
	pieces = []
	index = 2
	for i in range(m):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		pieces.append((a, b))
		
	R = set()
	C = set()
	S = set()
	T = set()
	
	for (a, b) in pieces:
		R.add(a)
		C.add(b)
		S.add(a + b)
		T.add(a - b)
		
	sizeA = len(R) * n
	sizeB = len(C) * n
	sizeC = 0
	for s_val in S:
		low_x = max(1, s_val - n)
		high_x = min(n, s_val - 1)
		if low_x <= high_x:
			sizeC += (high_x - low_x + 1)
			
	sizeD = 0
	for d_val in T:
		sizeD += (n - abs(d_val))
		
	AB = len(R) * len(C)
	AC = 0
	for x in R:
		for s_val in S:
			y = s_val - x
			if 1 <= y <= n:
				AC += 1
				
	AD = 0
	for x in R:
		for d_val in T:
			y = x - d_val
			if 1 <= y <= n:
				AD += 1
				
	BC = 0
	for y in C:
		for s_val in S:
			x = s_val - y
			if 1 <= x <= n:
				BC += 1
				
	BD = 0
	for y in C:
		for d_val in T:
			x = y + d_val
			if 1 <= x <= n:
				BD += 1
				
	CD = 0
	for s_val in S:
		for d_val in T:
			if (s_val + d_val) % 2 == 0:
				x = (s_val + d_val) // 2
				y = (s_val - d_val) // 2
				if 1 <= x <= n and 1 <= y <= n:
					CD += 1
					
	ABC = 0
	for x in R:
		for y in C:
			s_val = x + y
			if s_val in S:
				ABC += 1
				
	ABD = 0
	for x in R:
		for y in C:
			d_val = x - y
			if d_val in T:
				ABD += 1
				
	ACD = 0
	for x in R:
		for s_val in S:
			d_val = 2*x - s_val
			if d_val in T:
				if 1 <= s_val - x <= n:
					ACD += 1
					
	BCD = 0
	for y in C:
		for s_val in S:
			d_val = s_val - 2*y
			if d_val in T:
				if 1 <= s_val - y <= n:
					BCD += 1
					
	ABCD = 0
	for x in R:
		for y in C:
			s_val = x + y
			d_val = x - y
			if s_val in S and d_val in T:
				ABCD += 1
				
	total_union = sizeA + sizeB + sizeC + sizeD - (AB + AC + AD + BC + BD + CD) + (ABC + ABD + ACD + BCD) - ABCD
	safe_squares = n * n - total_union
	print(safe_squares)

if __name__ == "__main__":
	main()