def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+m]))
	total = [0] * m
	index = 2 + m
	for _ in range(n):
		foodnutrients = list(map(int, data[index:index+m]))
		index += m
		for j in range(m):
			total[j] += foodnutrients[j]
	for j in range(m):
		if total[j] < A[j]:
			print("No")
			return
	print("Yes")

if __name__ == '__main__':
	main()