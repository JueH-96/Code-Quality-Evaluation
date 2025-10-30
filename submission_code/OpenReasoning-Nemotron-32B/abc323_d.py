def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	groups = {}
	index = 1
	for _ in range(n):
		s = int(data[index])
		c = int(data[index + 1])
		index += 2
		k = s
		while k % 2 == 0:
			k //= 2
		total_val = c * (s // k)
		groups[k] = groups.get(k, 0) + total_val
	
	result = 0
	for V in groups.values():
		result += bin(V).count('1')
	
	print(result)

if __name__ == "__main__":
	main()