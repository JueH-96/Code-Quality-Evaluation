def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	N = int(data[0].strip())
	A = data[1].strip()
	total_length = 3 ** N
	
	arr = [1 if ch == '1' else 0 for ch in A]
	levels = N
	current = arr
	for _ in range(levels):
		next_level = []
		for i in range(0, len(current), 3):
			group = current[i:i+3]
			total = sum(group)
			if total >= 2:
				next_level.append(1)
			else:
				next_level.append(0)
		current = next_level
	original_root = current[0]
	
	base = []
	for char in A:
		if char == '0':
			base.append((0, 1))
		else:
			base.append((1, 0))
			
	for level in range(1, N+1):
		new_base = []
		for i in range(0, len(base), 3):
			left = base[i]
			mid = base[i+1]
			right = base[i+2]
			cost0 = min(
				left[0] + mid[0] + right[0],
				left[0] + mid[0] + right[1],
				left[0] + mid[1] + right[0],
				left[1] + mid[0] + right[0]
			)
			cost1 = min(
				left[1] + mid[1] + right[1],
				left[1] + mid[1] + right[0],
				left[1] + mid[0] + right[1],
				left[0] + mid[1] + right[1]
			)
			new_base.append((cost0, cost1))
		base = new_base
		
	root_state = base[0]
	if original_root == 0:
		answer = root_state[1]
	else:
		answer = root_state[0]
		
	print(answer)

if __name__ == "__main__":
	main()