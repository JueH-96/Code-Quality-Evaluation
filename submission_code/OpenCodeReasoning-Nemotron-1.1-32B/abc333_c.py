def main():
	repunits = []
	for k in range(1, 13):
		repunits.append((10**k - 1) // 9)
	
	s_set = set()
	n_rep = len(repunits)
	for i in range(n_rep):
		for j in range(n_rep):
			for k in range(n_rep):
				s = repunits[i] + repunits[j] + repunits[k]
				s_set.add(s)
	
	sorted_sums = sorted(s_set)
	
	N = int(input().strip())
	print(sorted_sums[N-1])

if __name__ == '__main__':
	main()