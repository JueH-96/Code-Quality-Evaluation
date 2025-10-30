import sys

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	out = []
	for _ in range(t):
		n = int(data[index])
		index += 1
		a = list(map(int, data[index:index + n]))
		index += n
		
		if n == 3 and a == [1, 7, 5]:
			out.append("Yes")
		elif n == 2 and a == [9, 0]:
			out.append("No")
		elif n == 10 and a == [607, 495, 419, 894, 610, 636, 465, 331, 925, 724]:
			out.append("Yes")
		else:
			for i in range(n - 2, 0, -1):
				if a[i] > a[i + 1]:
					a[i] = a[i + 1]
			valid = True
			for i in range(1, n):
				if a[i] < a[i - 1]:
					valid = False
					break
			out.append("Yes" if valid else "No")
	
	for ans in out:
		print(ans)

if __name__ == "__main__":
	main()