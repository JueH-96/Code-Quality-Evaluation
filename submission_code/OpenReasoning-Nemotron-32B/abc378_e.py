import sys

class Fenw:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, index, delta):
		i = index + 1
		while i <= self.n:
			self.tree[i] += delta
			i += i & -i
			
	def query(self, index):
		if index < 0:
			return 0
		s = 0
		i = index + 1
		while i:
			s += self.tree[i]
			i -= i & -i
		return s

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	M = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	total_subarray_sums = 0
	for i in range(n):
		total_subarray_sums += A[i] * (i + 1) * (n - i)
	
	S = [0] * (n + 1)
	for i in range(1, n + 1):
		S[i] = S[i - 1] + A[i - 1]
	
	Q = [0] * (n + 1)
	R = [0] * (n + 1)
	for i in range(n + 1):
		Q[i] = S[i] // M
		R[i] = S[i] % M
		
	part1 = 0
	part2 = 0
	for i in range(n + 1):
		part1 += i * Q[i]
		part2 += (n - i) * Q[i]
	T1 = part1 - part2
	
	fenw = Fenw(M)
	inv_count = 0
	for j in range(n + 1):
		r_val = R[j]
		count_le = fenw.query(r_val)
		count_gt = j - count_le
		inv_count += count_gt
		fenw.update(r_val, 1)
	
	T = T1 - inv_count
	ans = total_subarray_sums - M * T
	print(ans)

if __name__ == '__main__':
	main()