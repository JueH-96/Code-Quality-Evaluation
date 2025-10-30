import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if n == 0:
		print(0)
		return
		
	A.sort()
	max_val = A[-1]
	if max_val == 0:
		bits = 0
	else:
		bits = max_val.bit_length()
	
	if bits + 1 <= 20:
		up = 1 << (bits + 1)
	else:
		up = 10**6
		
	best_val = 0
	for i in range(0, up):
		count = 0
		for a in A:
			x = i + a
			ones = bin(x).count("1")
			if ones % 2 == 1:
				count += 1
		if count > best_val:
			best_val = count
			
	print(best_val)

if __name__ == '__main__':
	main()