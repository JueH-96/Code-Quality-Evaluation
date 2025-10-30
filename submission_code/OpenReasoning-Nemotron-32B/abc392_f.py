import sys

class SegmentTree:
	def __init__(self, n):
		self.n = n
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [0] * (2 * self.size)
	
	def build(self, arr):
		for i in range(self.n):
			self.data[self.size + i] = arr[i]
		for i in range(self.size - 1, 0, -1):
			self.data[i] = self.data[2*i] + self.data[2*i+1]
			
	def update(self, index, value):
		i = self.size + index
		self.data[i] = value
		while i > 1:
			i //= 2
			self.data[i] = self.data[2*i] + self.data[2*i+1]
			
	def query_kth(self, k):
		idx = 1
		l = 0
		r = self.size - 1
		while l < r:
			mid = (l + r) // 2
			if self.data[2*idx] >= k:
				idx = 2 * idx
				r = mid
			else:
				k -= self.data[2*idx]
				idx = 2 * idx + 1
				l = mid + 1
		return l

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	p = list(map(int, data[1:1+n]))
	
	seg = SegmentTree(n)
	arr = [1] * n
	seg.build(arr)
	res = [0] * n
	
	for i in range(n, 0, -1):
		k = p[i-1]
		pos = seg.query_kth(k)
		res[pos] = i
		seg.update(pos, 0)
		
	print(" ".join(map(str, res)))

if __name__ == "__main__":
	main()