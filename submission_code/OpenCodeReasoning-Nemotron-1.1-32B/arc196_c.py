MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	total_b = 0
	total_w = 0
	for char in s:
		if char == 'B':
			total_b += 1
		else:
			total_w += 1
		if total_w > total_b:
			print(0)
			return
	if total_b != n or total_w != n:
		print(0)
		return

	ans = 1
	b_count = 0
	for char in s:
		if char == 'B':
			b_count += 1
		else:
			ans = (ans * b_count) % MOD
			b_count -= 1
	print(ans % MOD)

if __name__ == '__main__':
	main()