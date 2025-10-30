import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0])
	s = data[1].strip()
	total = 0
	dp = 0
	for i in range(n):
		d = int(s[i])
		dp = dp * 10 + d * (i + 1)
		total += dp
	print(total)

if __name__ == '__main__':
	main()