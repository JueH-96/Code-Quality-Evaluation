import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	N = int(data[0])
	cum = 0
	d = 1
	while True:
		if d == 1:
			count_d = 10
		else:
			exp = (d - 1) // 2
			count_d = 9 * (10 ** exp)
		
		if N <= cum + count_d:
			k = N - cum
			if d == 1:
				res_str = str(k - 1)
			else:
				half_len = (d + 1) // 2
				start = 10 ** (half_len - 1)
				num = start + k - 1
				s = str(num)
				if d % 2 == 0:
					res_str = s + s[::-1]
				else:
					res_str = s + s[:-1][::-1]
			print(res_str)
			break
		else:
			cum += count_d
			d += 1

if __name__ == '__main__':
	main()