def main():
	points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
	dists = [3, 1, 4, 1, 5, 9]
	cumulative = [0]
	for d in dists:
		cumulative.append(cumulative[-1] + d)
	
	p, q = input().split()
	idx_p = points.index(p)
	idx_q = points.index(q)
	result = abs(cumulative[idx_p] - cumulative[idx_q])
	print(result)

if __name__ == "__main__":
	main()