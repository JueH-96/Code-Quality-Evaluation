import sys

class FenwickTree:
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
		i = index + 1
		s = 0
		while i:
			s += self.tree[i]
			i -= i & -i
		return s

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	M = int(data[1])
	A = list(map(int, data[2:2 + n]))
	
	P = [0] * (n + 1)
	for i in range(1, n + 1):
		P[i] = (P[i - 1] + A[i - 1]) % M
		
	S1 = 0
	S2 = 0
	for i in range(n + 1):
		S1 += i * P[i]
		S2 += P[i]
		
	tree = FenwickTree(M)
	inversion_count = 0
	for j in range(n + 1):
		cnt = tree.query(P[j])
		inversion_count += j - cnt
		tree.update(P[j], 1)
		
	total = (2 * S1 - n * S2) + M * inversion_count
	print(total)

if __name__ == "__main__":
	main()