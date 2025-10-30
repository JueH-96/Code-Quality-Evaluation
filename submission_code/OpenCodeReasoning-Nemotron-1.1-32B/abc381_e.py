import sys

INF = 10**18

class SegmentTreeMax:
	def __init__(self, n):
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [-INF] * (2 * self.size)
	
	def update(self, i, val):
		i += self.size
		self.data[i] = val
		while i > 1:
			i //= 2
			self.data[i] = max(self.data[2*i], self.data[2*i+1])
	
	def query(self, l, r):
		l += self.size
		r += self.size
		res = -INF
		while l <= r:
			if l % 2 == 1:
				res = max(res, self.data[l])
				l += 1
			if r % 2 == 0:
				res = max(res, self.data[r])
				r -= 1
			l //= 2
			r //= 2
		return res

class SegmentTreeMin:
	def __init__(self, n):
		self.size = 1
		while self.size < n:
			self.size *= 2
		self.data = [INF] * (2 * self.size)
	
	def update(self, i, val):
		i += self.size
		self.data[i] = val
		while i > 1:
			i //= 2
			self.data[i] = min(self.data[2*i], self.data[2*i+1])
	
	def query(self, l, r):
		l += self.size
		r += self.size
		res = INF
		while l <= r:
			if l % 2 == 1:
				res = min(res, self.data[l])
				l += 1
			if r % 2 == 0:
				res = min(res, self.data[r])
				r -= 1
			l //= 2
			r //= 2
		return res

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n, q = map(int, data[0].split())
	S = data[1].strip()
	
	P1 = [0] * (n+1)
	P2 = [0] * (n+1)
	for i in range(1, n+1):
		if S[i-1] == '1':
			P1[i] = P1[i-1] + 1
		else:
			P1[i] = P1[i-1]
		if S[i-1] == '2':
			P2[i] = P2[i-1] + 1
		else:
			P2[i] = P2[i-1]
	
	slashes = []
	for i in range(n):
		if S[i] == '/':
			key = P1[i] + P2[i+1]
			value1 = 2 * P1[i]
			value2 = P2[i+1]
			slashes.append((i, key, value1, value2))
	
	queries = []
	for i in range(2, 2+q):
		parts = data[i].split()
		L = int(parts[0])
		R = int(parts[1])
		l0 = L-1
		r0 = R-1
		T = P1[l0] + P2[r0+1]
		queries.append((l0, r0, T, i-2))
	
	queries1 = [(T, l0, r0, idx) for (l0, r0, T, idx) in queries]
	slashes1 = [(key, i, value1) for (i, key, value1, value2) in slashes]
	queries1_sorted = sorted(queries1, key=lambda x: x[0])
	slashes1_sorted = sorted(slashes1, key=lambda x: x[0])
	
	seg_tree_max = SegmentTreeMax(n)
	candidate1_arr = [-INF] * q
	
	ptr = 0
	for T_val, l0, r0, idx in queries1_sorted:
		while ptr < len(slashes1_sorted) and slashes1_sorted[ptr][0] <= T_val:
			i_index = slashes1_sorted[ptr][1]
			val = slashes1_sorted[ptr][2]
			seg_tree_max.update(i_index, val)
			ptr += 1
		res = seg_tree_max.query(l0, r0)
		candidate1_arr[idx] = res
	
	queries2 = [(T, l0, r0, idx) for (l0, r0, T, idx) in queries]
	slashes2 = [(key, i, value2) for (i, key, value1, value2) in slashes]
	queries2_sorted = sorted(queries2, key=lambda x: x[0], reverse=True)
	slashes2_sorted = sorted(slashes2, key=lambda x: x[0], reverse=True)
	
	seg_tree_min = SegmentTreeMin(n)
	candidate2_arr = [INF] * q
	
	ptr = 0
	for T_val, l0, r0, idx in queries2_sorted:
		while ptr < len(slashes2_sorted) and slashes2_sorted[ptr][0] > T_val:
			i_index = slashes2_sorted[ptr][1]
			val = slashes2_sorted[ptr][2]
			seg_tree_min.update(i_index, val)
			ptr += 1
		res = seg_tree_min.query(l0, r0)
		candidate2_arr[idx] = res
	
	out_lines = []
	for (l0, r0, T_val, idx) in queries:
		cand1 = candidate1_arr[idx]
		cand2 = candidate2_arr[idx]
		
		v1 = -INF
		if cand1 != -INF:
			v1 = cand1 - 2 * P1[l0] + 1
		
		v2 = -INF
		if cand2 != INF:
			v2 = -2 * cand2 + 2 * P2[r0+1] + 1
		
		ans = max(v1, v2, 0)
		out_lines.append(str(ans))
	
	print("
".join(out_lines))

if __name__ == '__main__':
	main()