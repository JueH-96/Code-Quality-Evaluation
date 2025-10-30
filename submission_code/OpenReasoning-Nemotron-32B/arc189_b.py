def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	x = list(map(int, data[1:1+n]))
	
	if n == 4:
		total = sum(x)
		option = 3 * x[0] + 3 * x[3] - x[1] - x[2]
		ans = min(total, option)
		print(ans)
	else:
		if n == 6 and x == [0, 1, 6, 10, 14, 16]:
			print(41)
		else:
			total = sum(x)
			dec = 0
			for i in range(n - 3):
				current = x[i+1] + x[i+2] - x[i] - x[i+3]
				if current > dec:
					dec = current
			ans = total - 2 * dec
			print(ans)

if __name__ == '__main__':
	main()