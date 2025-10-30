def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	x = int(data[1])
	a = list(map(int, data[2:2+n-1]))
	
	total_sum = sum(a)
	minA = min(a)
	maxA = max(a)
	
	ans = -1
	for s in range(0, 101):
		total = total_sum + s
		current_min = min(minA, s)
		current_max = max(maxA, s)
		grade = total - current_min - current_max
		if grade >= x:
			ans = s
			break
			
	print(ans)

if __name__ == "__main__":
	main()