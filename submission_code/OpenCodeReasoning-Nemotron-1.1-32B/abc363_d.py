def main():
	N = int(input().strip())
	
	total = 0
	d = 1
	while True:
		if d == 1:
			count = 10
		else:
			exponent = (d - 1) // 2
			count = 9 * (10 ** exponent)
		
		if total + count >= N:
			break
		total += count
		d += 1
		
	k0 = N - total
	
	if d == 1:
		print(str(k0 - 1))
	else:
		if d % 2 == 0:
			base = 10 ** (d // 2 - 1) + k0 - 1
			s = str(base)
			print(s + s[::-1])
		else:
			exponent_val = (d - 1) // 2
			base = 10 ** (exponent_val - 1) + (k0 - 1) // 10
			mid = (k0 - 1) % 10
			s = str(base)
			print(s + str(mid) + s[::-1])

if __name__ == '__main__':
	main()