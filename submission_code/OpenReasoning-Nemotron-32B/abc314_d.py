import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n = int(data[0].strip())
	s = data[1].strip()
	q = int(data[2].strip())
	
	arr = list(s)
	last_time = [0] * n
	
	global_op = 0
	global_time = 0
	
	operations = data[3:3+q]
	
	for idx, op in enumerate(operations):
		parts = op.split()
		t = int(parts[0])
		if t == 1:
			x = int(parts[1])
			c = parts[2]
			pos = x - 1
			arr[pos] = c
			last_time[pos] = idx + 1
		elif t == 2:
			global_op = 1
			global_time = idx + 1
		elif t == 3:
			global_op = 2
			global_time = idx + 1
			
	for i in range(n):
		if last_time[i] < global_time:
			if global_op == 1:
				arr[i] = arr[i].lower()
			elif global_op == 2:
				arr[i] = arr[i].upper()
				
	print(''.join(arr))

if __name__ == "__main__":
	main()