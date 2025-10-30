mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	N = int(data[0])
	M = int(data[1])
	ans = 0
	for i in range(0, 61):
		if M & (1 << i):
			total = N + 1
			cycle = 1 << (i + 1)
			full_cycles = total // cycle
			remainder = total % cycle
			count_i = full_cycles * (1 << i)
			if remainder > (1 << i):
				count_i += remainder - (1 << i)
			ans = (ans + count_i) % mod
	print(ans)

if __name__ == "__main__":
	main()