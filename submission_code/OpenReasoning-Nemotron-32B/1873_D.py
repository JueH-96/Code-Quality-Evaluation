t = int(input().strip())
results = []
for _ in range(t):
	n, k = map(int, input().split())
	s = input().strip()
	
	blacks = []
	for i, char in enumerate(s):
		if char == 'B':
			blacks.append(i)
			
	if not blacks:
		results.append(0)
		continue
		
	count = 0
	last_cover = -1
	for pos in blacks:
		if pos > last_cover:
			count += 1
			last_cover = pos + k - 1
			
	results.append(count)

for res in results:
	print(res)