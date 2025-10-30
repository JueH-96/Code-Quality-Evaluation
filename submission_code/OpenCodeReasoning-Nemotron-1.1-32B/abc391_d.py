import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	w = int(data[1])
	blocks = []
	index = 2
	for i in range(n):
		x = int(data[index])
		y = int(data[index+1])
		index += 2
		blocks.append((x, y, i+1))
	
	q = int(data[index])
	index += 1
	queries = []
	for i in range(q):
		t = int(data[index])
		a = int(data[index+1])
		index += 2
		queries.append((t, a, i))
	
	columns = [[] for _ in range(w+1)]
	for x, y, idx in blocks:
		if 1 <= x <= w:
			columns[x].append(y)
	
	common_min = None
	for col in range(1, w+1):
		if not columns[col]:
			common_min = None
			break
		if common_min is None:
			common_min = min(columns[col])
		else:
			common_min = min(common_min, min(columns[col]))
	
	if common_min is None:
		common_min = 10**18
	
	v_min = common_min
	results = []
	for t, a, q_idx in queries:
		x = blocks[a-1][0]
		y0 = blocks[a-1][1]
		if t < v_min:
			R_t = 0
		else:
			R_t = t - v_min + 1
		
		if y0 - 1 <= v_min:
			t0 = y0 - 1
		else:
			t0 = v_min + 1
		
		t_remove = max(t0, v_min)
		if t < t_remove:
			results.append("Yes")
		else:
			results.append("No")
	
	for res in results:
		print(res)

if __name__ == "__main__":
	main()