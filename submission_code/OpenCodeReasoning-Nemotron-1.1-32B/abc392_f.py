import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	P = list(map(int, data[1:1+n]))
	
	size = n
	tree = [0] * (4 * size)
	
	def build(node, l, r):
		if l == r:
			tree[node] = 1
		else:
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			build(left_child, l, mid)
			build(right_child, mid+1, r)
			tree[node] = tree[left_child] + tree[right_child]
	
	build(0, 0, n-1)
	
	def update(node, l, r, idx, val):
		if l == r:
			tree[node] = val
		else:
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			if idx <= mid:
				update(left_child, l, mid, idx, val)
			else:
				update(right_child, mid+1, r, idx, val)
			tree[node] = tree[left_child] + tree[right_child]
	
	def find_kth(k):
		node = 0
		l = 0
		r = n-1
		while l < r:
			mid = (l + r) // 2
			left_child = 2 * node + 1
			right_child = 2 * node + 2
			if tree[left_child] >= k:
				node = left_child
				r = mid
			else:
				k -= tree[left_child]
				node = right_child
				l = mid + 1
		return l

	arr = [0] * n
	for i in range(n-1, -1, -1):
		k = P[i]
		pos = find_kth(k)
		arr[pos] = i + 1
		update(0, 0, n-1, pos, 0)
		
	print(" ".join(map(str, arr)))

if __name__ == "__main__":
	main()