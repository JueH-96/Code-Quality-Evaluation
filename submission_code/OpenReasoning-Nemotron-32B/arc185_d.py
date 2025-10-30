mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	N = int(data[0])
	M = int(data[1])
	
	if N == 1 and M == 1:
		print(1)
	elif N == 1:
		ans = (M * M) % mod
		print(ans)
	elif M == 1:
		ans = (2 * N - 1) % mod
		print(ans)
	else:
		a = (N * M) % mod
		ans = (2 * M) % mod
		ans = ans * ((a + 1) % mod) % mod
		print(ans)

if __name__ == '__main__':
	main()