import sys

mod = 998244353

class Fenw:
	def __init__(self, size):
		self.n = size
		self.tree = [0] * (self.n + 1)
	
	def update(self, index, delta):
		while index <= self.n:
			self.tree[index] += delta
			index += index & -index
	
	def query(self, index):
		if index < 1:
			return 0
		s = 0
		while index:
			s += self.tree[index]
			index -= index & -index
		return s
	
	def range_query(self, l, r):
		if l > r:
			return 0
		return self.query(r) - self.query(l - 1)

def main():
	data = sys.stdin.read().split()
	if not data: 
		print(0)
		return
	it = iter(data)
	N = int(next(it))
	K = int(next(it))
	P = [int(next(it)) for _ in range(N)]
	
	fenw_inv = Fenw(N)
	total_inv_original = 0
	for i in range(N):
		if P[i] < N:
			c = fenw_inv.range_query(P[i] + 1, N)
		else:
			c = 0
		total_inv_original += c
		fenw_inv.update(P[i], 1)
	
	fenw_sum = Fenw(N)
	fenw_count = Fenw(N)
	total_sum_inside = 0
	
	for j in range(1, N + 1):
		if j - K >= 1:
			idx_remove = j - K - 1
			val_remove = P[idx_remove]
			fenw_sum.update(val_remove, -(j - K))
			fenw_count.update(val_remove, -1)
		
		if j >= 2:
			idx_add = j - 2
			val_add = P[idx_add]
			fenw_sum.update(val_add, j - 1)
			fenw_count.update(val_add, 1)
		
		current_val = P[j - 1]
		
		if current_val < N:
			s = fenw_sum.range_query(current_val + 1, N)
			c = fenw_count.range_query(current_val + 1, N)
		else:
			s = 0
			c = 0
			
		L0 = max(1, j - K + 1)
		S_j = s - (L0 - 1) * c
		total_sum_inside += S_j
		
	term1 = total_inv_original % mod
	term2 = (K * (K - 1)) % mod
	term2 = (term2 * pow(4, mod - 2, mod)) % mod
	term3 = total_sum_inside % mod
	term3 = (term3 * pow(N - K + 1, mod - 2, mod)) % mod
	
	ans = (term1 + term2 - term3) % mod
	if ans < 0:
		ans += mod
		
	print(ans)

if __name__ == '__main__':
	main()