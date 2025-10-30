def main():
	s = input().split()
	arr = list(map(int, s))
	
	freq = {}
	for a in arr:
		freq[a] = freq.get(a, 0) + 1
		
	three_candidates = []
	two_candidates = []
	for num, count in freq.items():
		if count >= 3:
			three_candidates.append(num)
		if count >= 2:
			two_candidates.append(num)
			
	if len(three_candidates) >= 2:
		print("Yes")
	elif len(three_candidates) == 1:
		if len(two_candidates) > 1:
			print("Yes")
		else:
			print("No")
	else:
		print("No")

if __name__ == '__main__':
	main()