import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	q = int(data[2].strip())
	
	base = list(s)
	char_set = base[:]
	last_update = [-1] * n
	
	ops = []
	index = 3
	for i in range(q):
		parts = data[index].split()
		index += 1
		t = int(parts[0])
		x = int(parts[1])
		c = parts[2]
		ops.append((t, x, c))
	
	last_global_in_sequence = None
	for i in range(q-1, -1, -1):
		if ops[i][0] in [2, 3]:
			last_global_in_sequence = (i, ops[i][0])
			break
	
	res = []
	for i in range(n):
		if last_update[i] == -1:
			c = base[i]
			last_global_op = last_global_in_sequence
		else:
			c = char_set[i]
			if last_global_in_sequence is not None and last_global_in_sequence[0] > last_update[i]:
				last_global_op = last_global_in_sequence
			else:
				last_global_op = None
		
		if last_global_op is None:
			res.append(c)
		else:
			t_op = last_global_op[1]
			if t_op == 2:
				res.append(c.lower())
			else:
				res.append(c.upper())
	
	print(''.join(res))

if __name__ == '__main__':
	main()