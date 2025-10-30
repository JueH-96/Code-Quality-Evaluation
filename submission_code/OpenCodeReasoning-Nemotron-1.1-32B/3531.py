class Solution:
	def minDamage(self, power: int, damage: List[int], health: List[int]) -> int:
		n = len(damage)
		if n == 0:
			return 0
		a_i = [(health[i] + power - 1) // power for i in range(n)]
		max_a = max(a_i) if a_i else 0
		max_t = max_a + n
		
		next_available = list(range(max_t + 2))
		
		def find_next(i):
			if next_available[i] != i:
				next_available[i] = find_next(next_available[i])
			return next_available[i]
		
		class Fenw:
			def __init__(self, size):
				self.n = size
				self.tree = [0] * (self.n + 1)
			
			def update(self, index, delta):
				while index <= self.n:
					self.tree[index] += delta
					index += index & -index
			
			def query(self, index):
				s = 0
				while index:
					s += self.tree[index]
					index -= index & -index
				return s
			
			def range_query(self, l, r):
				if l > r:
					return 0
				return self.query(r) - self.query(l - 1)
		
		fenw = Fenw(max_t)
		
		enemies = []
		for i in range(n):
			enemies.append((a_i[i], damage[i]))
		enemies.sort(key=lambda x: (x[1], x[0]))
		
		total_damage = 0
		for a, d in enemies:
			low = a
			high = max_t
			while low < high:
				mid = (low + high) // 2
				t0 = find_next(a)
				if t0 > mid:
					low = mid + 1
				else:
					count = fenw.range_query(1, t0 - 1)
					if count <= t0 - a:
						high = mid
					else:
						low = mid + 1
			t = find_next(a)
			while t <= max_t:
				count = fenw.range_query(1, t - 1)
				if count <= t - a:
					break
				t = find_next(t + 1)
			if t > max_t:
				t = find_next(a)
			total_damage += d * t
			fenw.update(t, 1)
			next_available[t] = find_next(t + 1)
		
		return total_damage