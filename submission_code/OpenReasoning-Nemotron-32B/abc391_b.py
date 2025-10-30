def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data:
		return
	
	n, m = map(int, data[0].split())
	S = data[1:1+n]
	T = data[1+n:1+n+m]
	
	for a in range(n - m + 1):
		for b in range(n - m + 1):
			match = True
			for i in range(m):
				if S[a+i][b:b+m] != T[i]:
					match = False
					break
			if match:
				print(f"{a+1} {b+1}")
				return

if __name__ == "__main__":
	main()