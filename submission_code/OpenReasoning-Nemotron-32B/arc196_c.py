mod = 998244353

def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0])
	s = data[1].strip()
	
	if s[0] != 'B' or s[-1] != 'W':
		print(0)
		return
		
	inner = s[1:-1]
	ans = 1
	open_count = 0
	
	for c in inner:
		if c == 'W':
			open_count += 1
		else:
			if open_count <= 0:
				ans = 0
			else:
				ans = (ans * open_count) % mod
			open_count -= 1
			
	print(ans % mod)

if __name__ == '__main__':
	main()