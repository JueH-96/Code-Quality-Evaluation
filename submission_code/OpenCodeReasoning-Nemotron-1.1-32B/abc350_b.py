def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	q = int(data[1])
	treatments = list(map(int, data[2:2+q]))
	
	freq = [0] * (n + 1)
	for t in treatments:
		freq[t] += 1
			
	count_odd = sum(freq[i] % 2 for i in range(1, n + 1))
	result = n - count_odd
	print(result)

if __name__ == "__main__":
	main()