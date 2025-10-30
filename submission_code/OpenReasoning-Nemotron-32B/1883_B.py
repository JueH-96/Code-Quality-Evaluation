t = int(input().strip())
results = []
for _ in range(t):
	n, k = map(int, input().split())
	s = input().strip()
	
	freq = {}
	for char in s:
		freq[char] = freq.get(char, 0) + 1
		
	odd_count = 0
	total_pairs = 0
	for count in freq.values():
		if count % 2 == 1:
			odd_count += 1
		total_pairs += count // 2
		
	if odd_count == 0:
		min_removals1 = 1
		total_pairs_after1 = total_pairs - 1
	else:
		min_removals1 = odd_count - 1
		total_pairs_after1 = total_pairs
		
	if odd_count == 0:
		min_removals2 = 0
		total_pairs_after2 = total_pairs
	else:
		min_removals2 = odd_count
		total_pairs_after2 = total_pairs
		
	cond1 = False
	if k >= min_removals1:
		extra = k - min_removals1
		if extra % 2 == 0 and extra <= 2 * total_pairs_after1:
			cond1 = True
			
	cond2 = False
	if k >= min_removals2:
		extra = k - min_removals2
		if extra % 2 == 0 and extra <= 2 * total_pairs_after2:
			cond2 = True
			
	if cond1 or cond2:
		results.append("YES")
	else:
		results.append("NO")
		
for res in results:
	print(res)