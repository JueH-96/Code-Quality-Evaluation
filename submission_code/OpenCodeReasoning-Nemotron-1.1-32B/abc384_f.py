import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	diag = [0] * 26
	
	for x in A:
		v = 0
		temp = x
		while temp % 2 == 0:
			v += 1
			temp //= 2
		if v < 26:
			k0 = v + 1
			if k0 < 26:
				diag[k0] += x >> v
				
	ans = 0
	for k in range(0, 26):
		M = 1 << (k + 1)
		target = 1 << k
		freq = {}
		sum_a = {}
		for x in A:
			r = x % M
			a_val = x >> (k + 1)
			freq[r] = freq.get(r, 0) + 1
			sum_a[r] = sum_a.get(r, 0) + a_val
			
		total_value = 0
		for r_i in freq:
			r_j = (target - r_i) % M
			if r_j not in freq:
				continue
			count_i = freq[r_i]
			count_j = freq[r_j]
			sum_i = sum_a[r_i]
			sum_j = sum_a[r_j]
			
			if r_i < r_j:
				if r_i <= target:
					category = 1
				else:
					category = 2
				part1 = 2 * (sum_i * count_j + sum_j * count_i)
				part2 = (1 if category == 1 else 3) * (count_i * count_j)
				total_value += part1 + part2
			elif r_i == r_j:
				if r_i <= target:
					category = 1
				else:
					category = 2
				part1 = 2 * (count_i - 1) * sum_i
				part2 = (1 if category == 1 else 3) * (count_i * (count_i - 1) // 2)
				total_value += part1 + part2
				
		if 1 <= k < 26:
			total_value += diag[k]
			
		ans += total_value
		
	print(ans)

if __name__ == "__main__":
	main()