import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	P = [0] * (n + 1)
	for i in range(1, n + 1):
		P[i] = P[i-1] ^ A[i-1]
		
	total_ans = 0
	for k in range(32):
		count0 = 0
		count1 = 0
		subtract = 0
		prev_bit = None
		for i in range(n + 1):
			bit = (P[i] >> k) & 1
			if prev_bit is not None:
				if bit != prev_bit:
					subtract += 1
			if bit:
				count1 += 1
			else:
				count0 += 1
			prev_bit = bit
			
		total_pairs = count0 * count1
		count_k = total_pairs - subtract
		total_ans += (1 << k) * count_k
		
	print(total_ans)

if __name__ == "__main__":
	main()