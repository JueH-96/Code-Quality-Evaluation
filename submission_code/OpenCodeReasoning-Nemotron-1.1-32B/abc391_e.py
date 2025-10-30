def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	
	current = s
	for _ in range(n):
		next_s = []
		for j in range(0, len(current), 3):
			group = current[j:j+3]
			count0 = group.count('0')
			if count0 >= 2:
				next_s.append('0')
			else:
				next_s.append('1')
		current = ''.join(next_s)
	
	original_root = current
	
	dp = []
	for char in s:
		if char == '0':
			dp.append((0, 1))
		else:
			dp.append((1, 0))
			
	for _ in range(n):
		new_dp = []
		for j in range(0, len(dp), 3):
			l0, l1 = dp[j]
			m0, m1 = dp[j+1]
			r0, r1 = dp[j+2]
			
			min_r = min(r0, r1)
			min_m = min(m0, m1)
			min_l = min(l0, l1)
			
			option1 = l0 + m0 + min_r
			option2 = l0 + r0 + min_m
			option3 = m0 + r0 + min_l
			node0 = min(option1, option2, option3)
			
			option1 = l1 + m1 + min_r
			option2 = l1 + r1 + min_m
			option3 = m1 + r1 + min_l
			node1 = min(option1, option2, option3)
			
			new_dp.append((node0, node1))
		dp = new_dp
		
	root0, root1 = dp[0]
	if original_root == '0':
		print(root1)
	else:
		print(root0)

if __name__ == "__main__":
	main()