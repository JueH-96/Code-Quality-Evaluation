import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+m]))
	
	j = m - 1
	last_fire = A[j]
	ans = [0] * (n + 1)
	
	for i in range(n, 0, -1):
		if j >= 0 and i == A[j]:
			ans[i] = 0
			last_fire = i
			j -= 1
		else:
			ans[i] = last_fire - i
	
	for i in range(1, n + 1):
		print(ans[i])

if __name__ == "__main__":
	main()