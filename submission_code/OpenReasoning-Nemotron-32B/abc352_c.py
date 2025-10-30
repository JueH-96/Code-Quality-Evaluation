def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	giants = []
	total_A = 0
	index = 1
	for i in range(n):
		a = int(data[index])
		b = int(data[index + 1])
		index += 2
		giants.append((a, b))
		total_A += a
	
	ans = 0
	for a, b in giants:
		candidate = total_A - a + b
		if candidate > ans:
			ans = candidate
	
	print(ans)

if __name__ == "__main__":
	main()