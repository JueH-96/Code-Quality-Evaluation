import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+m]))
	
	is_firework = [False] * (n + 1)
	for a in A:
		is_firework[a] = True
		
	res = [0] * (n + 1)
	last = 10**9
	
	for day in range(n, 0, -1):
		if is_firework[day]:
			res[day] = 0
			last = day
		else:
			res[day] = last - day
			
	for day in range(1, n + 1):
		print(res[day])

if __name__ == '__main__':
	main()