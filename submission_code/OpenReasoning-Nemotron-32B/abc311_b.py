def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	n, d = map(int, data[0].split())
	schedules = []
	for i in range(1, 1 + n):
		s = data[i].strip()
		schedules.append(s)
	
	free_day = [True] * d
	for j in range(d):
		for i in range(n):
			if schedules[i][j] != 'o':
				free_day[j] = False
				break
	
	max_consecutive = 0
	current = 0
	for j in range(d):
		if free_day[j]:
			current += 1
		else:
			current = 0
		if current > max_consecutive:
			max_consecutive = current
	
	print(max_consecutive)

if __name__ == "__main__":
	main()