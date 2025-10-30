import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	parts = data[0].split()
	if not parts:
		print(0)
		return
	N = int(parts[0])
	M = int(parts[1])
	K = int(parts[2])
	
	tests = []
	for i in range(1, M+1):
		line = data[i].split()
		if not line:
			continue
		c_i = int(line[0])
		keys = list(map(int, line[1:1+c_i]))
		r_i = line[1+c_i]
		test_mask = 0
		for key in keys:
			test_mask |= (1 << (key-1))
		tests.append((test_mask, r_i))
	
	max_mask = 1 << N
	popcount_arr = [bin(i).count('1') for i in range(max_mask)]
	
	total_valid = 0
	for mask in range(max_mask):
		valid = True
		for (t_mask, r) in tests:
			combined = mask & t_mask
			count = popcount_arr[combined]
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

if __name__ == '__main__':
	main()