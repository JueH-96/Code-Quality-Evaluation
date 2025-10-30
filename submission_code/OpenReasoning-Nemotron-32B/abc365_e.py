import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	
	total_single = sum(arr)
	
	prefix = [0] * (n+1)
	for i in range(1, n+1):
		prefix[i] = prefix[i-1] ^ arr[i-1]
	
	num_bits = 31
	total_T = 0
	
	for b in range(num_bits):
		count0 = 0
		count1 = 0
		total_pairs = 0
		for p in prefix:
			if (p >> b) & 1:
				total_pairs += count0
				count1 += 1
			else:
				total_pairs += count1
				count0 += 1
		total_T += (1 << b) * total_pairs
		
	answer = total_T - total_single
	print(answer)

if __name__ == "__main__":
	main()