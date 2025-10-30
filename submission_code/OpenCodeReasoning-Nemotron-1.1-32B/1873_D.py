t = int(input().strip())
for _ in range(t):
	data = input().split()
	n = int(data[0])
	k = int(data[1])
	s = input().strip()
	
	if 'B' not in s:
		print(0)
		continue
		
	covered_until = -1
	ans = 0
	for i in range(n):
		if s[i] == 'B':
			if i > covered_until:
				ans += 1
				start = min(i, n - k)
				covered_until = start + k - 1
				
	print(ans)