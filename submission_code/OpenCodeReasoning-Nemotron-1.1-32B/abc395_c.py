def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	min_len = float('inf')
	last_occurrence = {}
	
	for i in range(n):
		x = A[i]
		if x in last_occurrence:
			gap = i - last_occurrence[x] + 1
			if gap < min_len:
				min_len = gap
			last_occurrence[x] = i
		else:
			last_occurrence[x] = i
			
	if min_len == float('inf'):
		print(-1)
	else:
		print(min_len)

if __name__ == "__main__":
	main()