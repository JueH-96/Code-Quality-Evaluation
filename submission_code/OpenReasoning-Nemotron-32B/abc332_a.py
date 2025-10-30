def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	S = int(data[1])
	K = int(data[2])
	total = 0
	index = 3
	for i in range(n):
		p = int(data[index])
		q = int(data[index+1])
		index += 2
		total += p * q
	if total >= S:
		print(total)
	else:
		print(total + K)

if __name__ == "__main__":
	main()