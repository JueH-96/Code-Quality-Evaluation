import sys

class FenwickTree:
	def __init__(self, n):
		self.n = n
		self.tree = [0] * (n + 1)
	
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

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	
	n = int(data[0])
	chords = []
	idx_data = 1
	for i in range(n):
		a = int(data[idx_data])
		b = int(data[idx_data + 1])
		idx_data += 2
		if a > b:
			a, b = b, a
		chords.append((a, b))
	
	size = 2 * n

	list2 = []
	for i, (a, b) in enumerate(chords):
		list2.append((b, a, i))
	list2.sort(key=lambda x: x[0])

	fenw_f = FenwickTree(size)
	F = [0] * n

	for b_val, a_val, i in list2:
		if a_val + 1 <= b_val - 1:
			cnt = fenw_f.range_query(a_val + 1, b_val - 1)
		else:
			cnt = 0
		F[i] = cnt
		fenw_f.update(a_val, 1)

	left_present = [False] * (size + 1)
	for a, b in chords:
		left_present[a] = True

	list3 = []
	for i, (a, b) in enumerate(chords):
		list3.append((b, a, i))
	list3.sort(key=lambda x: x[0])

	fenw_A = FenwickTree(size)
	A_val = [0] * n

	j = 0
	for coord in range(1, size + 1):
		if coord <= size and left_present[coord]:
			fenw_A.update(coord, 1)
		
		while j < len(list3) and list3[j][0] == coord:
			b_val, a_val, i = list3[j]
			if a_val + 1 <= b_val - 1:
				cntA = fenw_A.range_query(a_val + 1, b_val - 1)
			else:
				cntA = 0
			A_val[i] = cntA
			j += 1

	right_present = [False] * (size + 1)
	for a, b in chords:
		right_present[b] = True

	list4 = []
	for i, (a, b) in enumerate(chords):
		list4.append((b, a, i))
	list4.sort(key=lambda x: x[0])

	fenw_B = FenwickTree(size)
	B_val = [0] * n

	j = 0
	for coord in range(1, size + 1):
		if coord <= size and right_present[coord]:
			fenw_B.update(coord, 1)
		
		while j < len(list4) and list4[j][0] == coord:
			b_val, a_val, i = list4[j]
			if a_val + 1 <= b_val - 1:
				cntB = fenw_B.range_query(a_val + 1, b_val - 1)
			else:
				cntB = 0
			B_val[i] = cntB
			j += 1

	for i in range(n):
		if A_val[i] + B_val[i] - 2 * F[i] > 0:
			print("Yes")
			return
	
	print("No")

if __name__ == "__main__":
	main()