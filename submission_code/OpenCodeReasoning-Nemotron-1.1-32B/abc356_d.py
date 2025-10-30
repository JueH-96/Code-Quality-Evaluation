mod = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	N = int(data[0])
	M = int(data[1])
	
	total = 0
	for b in range(0, 61):
		if M & (1 << b):
			cycle = 1 << (b + 1)
			full_cycles = (N + 1) // cycle
			count_b = full_cycles * (1 << b)
			rem = (N + 1) % cycle
			if rem > (1 << b):
				count_b += rem - (1 << b)
			total = (total + count_b) % mod
			
	print(total)

if __name__ == "__main__":
	main()