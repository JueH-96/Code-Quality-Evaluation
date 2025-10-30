import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	t = int(data[1])
	A = list(map(int, data[2:2+t]))
	
	rows = [0] * n
	cols = [0] * n
	diag1 = 0
	diag2 = 0
	
	for turn in range(t):
		a = A[turn]
		i = (a - 1) // n
		j = (a - 1) % n
		
		rows[i] += 1
		cols[j] += 1
		
		if i == j:
			diag1 += 1
		if j == n - 1 - i:
			diag2 += 1
			
		if rows[i] == n or cols[j] == n or diag1 == n or diag2 == n:
			print(turn + 1)
			return
			
	print(-1)

if __name__ == "__main__":
	main()