import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return

	first_line = data[0].split()
	N = int(first_line[0])
	M = int(first_line[1])
	K = int(first_line[2])
	
	total_masks = 1 << N
	popcount_arr = [0] * total_masks
	for mask in range(total_masks):
		popcount_arr[mask] = bin(mask).count("1")
	
	tests = []
	for i in range(1, M+1):
		parts = data[i].split()
		c_i = int(parts[0])
		keys = list(map(int, parts[1:1+c_i]))
		r_i = parts[1+c_i]
		bitmask = 0
		for key in keys:
			bitmask |= (1 << (key-1))
		tests.append((bitmask, r_i))
	
	total_valid = 0
	for assignment in range(total_masks):
		valid = True
		for (bitmask, r) in tests:
			count = popcount_arr[assignment & bitmask]
			if r == 'o':
				if count < K:
					valid = False
					break
			else:
				if count >= K:
					valid = False
					break
		if valid:
			total_valid += 1
			
	print(total_valid)

if __name__ == "__main__":
	main()