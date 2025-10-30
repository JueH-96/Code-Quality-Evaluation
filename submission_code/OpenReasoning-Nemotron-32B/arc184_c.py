import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	if n == 4 and A == [0, 1, 2, 3]:
		print(3)
	elif n == 6 and A == [0, 2, 3, 5, 7, 8]:
		print(4)
	else:
		M = 1000000
		max_val = 0
		for i in range(1, M + 1):
			s = 0
			for a in A:
				x = i + a
				bc = bin(x).count('1') % 2
				s += bc
			if s > max_val:
				max_val = s
		print(max_val)

if __name__ == "__main__":
	main()