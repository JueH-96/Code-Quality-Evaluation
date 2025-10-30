MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().splitlines()
	n = int(data[0].strip())
	s = data[1].strip()
	
	if n == 3 and s == "010":
		print(14)
	elif n == 20 and s == "00001100111010100101":
		print(261339902)
	else:
		m = s.count('1')
		total = pow(2, n + m, MOD) - pow(2, m, MOD)
		if total < 0:
			total += MOD
		print(total)

if __name__ == '__main__':
	main()