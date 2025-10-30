import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	m = int(data[1])
	A = []
	B = []
	C = []
	idx = 2
	for i in range(n):
		t = int(data[idx])
		x = int(data[idx+1])
		idx += 2
		if t == 0:
			A.append(x)
		elif t == 1:
			B.append(x)
		else:
			C.append(x)
			
	A.sort(reverse=True)
	B.sort(reverse=True)
	C.sort(reverse=True)
	
	prefixA = [0]
	for val in A:
		prefixA.append(prefixA[-1] + val)
	
	prefixB = [0]
	for val in B:
		prefixB.append(prefixB[-1] + val)
		
	prefixC = [0]
	for val in C:
		prefixC.append(prefixC[-1] + val)
		
	best_ans = 0
	max_c = min(m, len(C))
	
	for c in range(0, max_c + 1):
		k = m - c
		if k < 0:
			break
			
		cap = prefixC[c]
		b_max = min(len(B), k, cap)
		b_low = max(0, k - len(A))
		
		if b_low > b_max:
			continue
			
		l = b_low
		r = b_max
		
		while r - l > 2:
			m1 = l + (r - l) // 3
			m2 = r - (r - l) // 3
			f1 = prefixB[m1] + prefixA[k - m1]
			f2 = prefixB[m2] + prefixA[k - m2]
			if f1 < f2:
				l = m1 + 1
			else:
				r = m2 - 1
				
		res_c = 0
		for b0 in range(l, r + 1):
			total_val = prefixB[b0] + prefixA[k - b0]
			if total_val > res_c:
				res_c = total_val
				
		if res_c > best_ans:
			best_ans = res_c
			
	print(best_ans)

if __name__ == "__main__":
	main()