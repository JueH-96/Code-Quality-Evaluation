import sys

INF_val = 10**30
INF_float = float('inf')

class Node:
	__slots__ = ('sum', 'prod', 'function')
	def __init__(self, sum_val, prod_val, function):
		self.sum = sum_val
		self.prod = prod_val
		self.function = function

def create_leaf(A_i, B_i):
	if B_i == 1:
		function = [ (-INF_float, INF_float, 1, A_i) ]
	else:
		threshold = A_i / (B_i - 1.0)
		function = [ (-INF_float, threshold, 1, A_i), (threshold, INF_float, B_i, 0) ]
	return Node(A_i, B_i, function)

def compose_functions(f, g):
	pieces = []
	for (L_f, R_f, a, b) in f:
		for (L_g, R_g, c, d) in g:
			if a == 0:
				if L_g <= b <= R_g:
					C_low = L_f
					C_high = R_f
				else:
					continue
			else:
				if a > 0:
					C_low = max(L_f, (L_g - b) / a)
					C_high = min(R_f, (R_g - b) / a)
				else:
					continue
			if C_low > C_high:
				continue
			slope_new = a * c
			intercept_new = b * c + d
			pieces.append( (C_low, C_high, slope_new, intercept_new) )
	
	if not pieces:
		return [ (-INF_float, INF_float, 0, 0) ]
	
	breakpoints = set()
	for (C_low, C_high, slope, intercept) in pieces:
		breakpoints.add(C_low)
		breakpoints.add(C_high)
	breakpoints = sorted(breakpoints)
	new_pieces = []
	
	if breakpoints and breakpoints[0] > -INF_float:
		mid = breakpoints[0] - 1.0
		best_val = -10**30
		best_piece = None
		for piece in pieces:
			if piece[0] <= mid <= piece[1]:
				val = piece[2] * mid + piece[3]
				if val > best_val:
					best_val = val
					best_piece = piece
		if best_piece is not None:
			new_pieces.append( (-INF_float, breakpoints[0], best_piece[2], best_piece[3]) )
		else:
			new_pieces.append( (-INF_float, breakpoints[0], 0, 0) )
	
	for i in range(len(breakpoints)-1):
		mid = (breakpoints[i] + breakpoints[i+1]) / 2.0
		best_val = -10**30
		best_piece = None
		for piece in pieces:
			if piece[0] <= mid <= piece[1]:
				val = piece[2] * mid + piece[3]
				if val > best_val:
					best_val = val
					best_piece = piece
		if best_piece is not None:
			new_pieces.append( (breakpoints[i], breakpoints[i+1], best_piece[2], best_piece[3]) )
		else:
			new_pieces.append( (breakpoints[i], breakpoints[i+1], 0, 0) )
	
	if breakpoints and breakpoints[-1] < INF_float:
		mid = breakpoints[-1] + 1.0
		best_val = -10**30
		best_piece = None
		for piece in pieces:
			if piece[0] <= mid <= piece[1]:
				val = piece[2] * mid + piece[3]
				if val > best_val:
					best_val = val
					best_piece = piece
		if best_piece is not None:
			new_pieces.append( (breakpoints[-1], INF_float, best_piece[2], best_piece[3]) )
		else:
			new_pieces.append( (breakpoints[-1], INF_float, 0, 0) )
	
	merged = []
	for piece in new_pieces:
		if not merged:
			merged.append(piece)
		else:
			last = merged[-1]
			if last[2] == piece[2] and last[3] == piece[3] and last[1] == piece[0]:
				merged[-1] = (last[0], piece[1], last[2], last[3])
			else:
				merged.append(piece)
	return merged

def combine_nodes(left, right):
	sum_val = left.sum + right.sum
	prod_val = left.prod * right.prod
	composed_function = compose_functions(left.function, right.function)
	return Node(sum_val, prod_val, composed_function)

class SegmentTree:
	def __init__(self, A, B):
		self.n = len(A)
		self.A = A
		self.B = B
		self.size = 1
		while self.size < self.n:
			self.size *= 2
		self.tree = [None] * (2 * self.size)
		self.build(0, 0, self.n-1)
	
	def build(self, node, l, r):
		if l == r:
			self.tree[node] = create_leaf(self.A[l], self.B[l])
		else:
			mid = (l + r) // 2
			self.build(2*node+1, l, mid)
			self.build(2*node+2, mid+1, r)
			left_child = self.tree[2*node+1]
			right_child = self.tree[2*node+2]
			self.tree[node] = combine_nodes(left_child, right_child)
	
	def update(self, node, l, r, index, A_val, B_val):
		if l == r:
			self.tree[node] = create_leaf(A_val, B_val)
		else:
			mid = (l + r) // 2
			if index <= mid:
				self.update(2*node+1, l, mid, index, A_val, B_val)
			else:
				self.update(2*node+2, mid+1, r, index, A_val, B_val)
			left_child = self.tree[2*node+1]
			right_child = self.tree[2*node+2]
			self.tree[node] = combine_nodes(left_child, right_child)
	
	def update_point(self, index, A_val, B_val):
		self.update(0, 0, self.n-1, index, A_val, B_val)
	
	def query_range(self, node, l, r, ql, qr):
		if qr < l or r < ql:
			return None
		if ql <= l and r <= qr:
			return self.tree[node]
		mid = (l + r) // 2
		left_res = self.query_range(2*node+1, l, mid, ql, qr)
		right_res = self.query_range(2*node+2, mid+1, r, ql, qr)
		if left_res is None:
			return right_res
		if right_res is None:
			return left_res
		return combine_nodes(left_res, right_res)
	
	def query(self, ql, qr):
		return self.query_range(0, 0, self.n-1, ql, qr)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	B = list(map(int, data[1+n:1+2*n]))
	q = int(data[1+2*n])
	queries = []
	index = 1+2*n+1
	for i in range(q):
		t = data[index]; index += 1
		if t == '1' or t == '2':
			i_index = int(data[index]); index += 1
			x = int(data[index]); index += 1
			queries.append( (t, i_index, x) )
		else:
			l = int(data[index]); index += 1
			r = int(data[index]); index += 1
			queries.append( (t, l, r) )
	
	seg_tree = SegmentTree(A, B)
	output_lines = []
	for query_item in queries:
		if query_item[0] == '1':
		 typ, i_index, x = query_item
			i_index -= 1
			seg_tree.update_point(i_index, x, seg_tree.B[i_index])
		elif query_item[0] == '2':
			typ, i_index, x = query_item
			i_index -= 1
			seg_tree.update_point(i_index, seg_tree.A[i_index], x)
		else:
			typ, l, r = query_item
			l -= 1; r -= 1
			node = seg_tree.query(l, r)
			best_val = -10**30
			for (low, high, slope, intercept) in node.function:
				if low <= 0 <= high:
					val = slope * 0 + intercept
					if val > best_val:
						best_val = val
			output_lines.append(str(int(best_val)))
	
	print("
".join(output_lines))

if __name__ == '__main__':
	main()