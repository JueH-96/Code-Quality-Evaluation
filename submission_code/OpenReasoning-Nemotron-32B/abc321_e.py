def count_in_subtree(root, d, N):
	if d < 0:
		return 0
	if root > N:
		return 0
	max_d = N.bit_length() - root.bit_length()
	if d > max_d:
		return 0
	low = root << d
	if low > N:
		return 0
	num_nodes = 1 << d
	high = low + num_nodes - 1
	if high > N:
		return N - low + 1
	else:
		return num_nodes

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index])
		X = int(data[index+1])
		K = int(data[index+2])
		index += 3
		count1 = count_in_subtree(X, K, N)
		count2 = 0
		a = 1
		current = X
		while current != 1 and a <= K:
			parent = current // 2
			if a == K:
				count2 += 1
			if current == 2 * parent:
				sibling = 2 * parent + 1
			else:
				sibling = 2 * parent
			if sibling <= N:
				rem = K - a - 1
				if rem >= 0:
					count2 += count_in_subtree(sibling, rem, N)
			current = parent
			a += 1
		results.append(str(count1 + count2))
	print("
".join(results))

if __name__ == "__main__":
	main()