import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	points = []
	for i in range(1, n + 1):
		if not data[i].strip():
			continue
		x, y = map(int, data[i].split())
		points.append((x, y))
	
	group0 = []
	group1 = []
	
	for (x, y) in points:
		if (x + y) % 2 == 0:
			group0.append((x, y))
		else:
			group1.append((x, y))
			
	total_ans = 0
	
	for group in [group0, group1]:
		n_group = len(group)
		if n_group < 2:
			continue
			
		u_list = []
		v_list = []
		for (x, y) in group:
			u_list.append(x + y)
			v_list.append(x - y)
			
		u_list.sort()
		v_list.sort()
		
		S_u = 0
		for i in range(n_group):
			S_u += u_list[i] * (2 * i - n_group + 1)
			
		S_v = 0
		for i in range(n_group):
			S_v += v_list[i] * (2 * i - n_group + 1)
			
		total_ans += (S_u + S_v) // 2
		
	print(total_ans)

if __name__ == "__main__":
	main()