import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	results = []
	for _ in range(t):
		N = int(data[index])
		index += 1
		A = list(map(int, data[index:index + 2 * N]))
		index += 2 * N

		n = 2 * N
		if n == 0:
			results.append("0")
			continue

		forward_dict = {}
		backward_dict = {}
		total = 0
		for i in range(n - 1):
			if i >= 2:
				a1 = A[i - 2]
				a2 = A[i - 1]
				if a1 != a2:
					key = (a1, a2)
					forward_dict[key] = forward_dict.get(key, 0) + 1

			if i >= 3:
				a1 = A[i - 3]
				a2 = A[i - 2]
				if a1 != a2:
					key = (a1, a2)
					backward_dict[key] = backward_dict.get(key, 0) + 1

			x = A[i]
			y = A[i + 1]
			if x == y:
				continue

			count1 = forward_dict.get((x, y), 0)
			count2 = backward_dict.get((y, x), 0)
			total += count1 + count2

		results.append(str(total))
	
	print("
".join(results))

if __name__ == '__main__':
	main()