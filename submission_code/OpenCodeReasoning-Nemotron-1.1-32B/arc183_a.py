import sys

def main():
	data = sys.stdin.read().split()
	N = int(data[0])
	K = int(data[1])
	total_length = N * K
	
	if total_length == 0:
		print("")
		return
		
	total = 1
	for i in range(1, total_length + 1):
		total *= i
		
	for i in range(1, K + 1):
		for j in range(N):
			total //= i
			
	k0 = (total + 1) // 2 - 1
	counts = [K] * N
	T = total_length
	current_total = total
	k = k0
	res = []
	
	for _ in range(total_length):
		for i in range(N):
			if counts[i] == 0:
				continue
			count_i = current_total * counts[i] // T
			if k < count_i:
				res.append(i + 1)
				current_total = count_i
				counts[i] -= 1
				T -= 1
				break
			else:
				k -= count_i
				
	print(" ".join(map(str, res)))
	
if __name__ == '__main__':
	main()