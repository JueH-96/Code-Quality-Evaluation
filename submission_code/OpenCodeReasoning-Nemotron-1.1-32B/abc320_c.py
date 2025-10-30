def main():
	M = int(input().strip())
	S1 = input().strip()
	S2 = input().strip()
	S3 = input().strip()
	
	best = 10**9
	
	for d in '0123456789':
		R1 = [i for i in range(M) if S1[i] == d]
		R2 = [i for i in range(M) if S2[i] == d]
		R3 = [i for i in range(M) if S3[i] == d]
		if not R1 or not R2 or not R3:
			continue
			
		for r1 in R1:
			for r2 in R2:
				for r3 in R3:
					arr = sorted([r1, r2, r3])
					seen = set()
					max_time = 0
					for x in arr:
						t_val = x
						while t_val in seen:
							t_val += M
						seen.add(t_val)
						if t_val > max_time:
							max_time = t_val
					if max_time < best:
						best = max_time
						
	if best == 10**9:
		print(-1)
	else:
		print(best)

if __name__ == "__main__":
	main()