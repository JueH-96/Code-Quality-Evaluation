def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	
	n, d = map(int, data[0].split())
	schedules = []
	for i in range(1, 1 + n):
		schedules.append(data[i].strip())
	
	valid = [True] * d
	for j in range(d):
		for i in range(n):
			if schedules[i][j] == 'x':
				valid[j] = False
				break
	
	max_days = 0
	current = 0
	for j in range(d):
		if valid[j]:
			current += 1
			if current > max_days:
				max_days = current
		else:
			current = 0
			
	print(max_days)

if __name__ == "__main__":
	main()