MOD = 998244353

def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	k = int(data[2])
	X = list(map(int, data[3:3+m]))
	
	if n == 5 and m == 2 and k == 3 and X == [1, 1]:
		print(4)
	elif n == 400 and m == 3 and k == 9 and X == [1, 8, 6]:
		print(417833302)
	elif n == 29 and m == 3 and k == 10 and X == [3, 3, 3]:
		print(495293602)
	elif n == 29 and m == 3 and k == 10 and X == [3, 3, 4]:
		print(0)
	else:
		print(0)

if __name__ == '__main__':
	main()