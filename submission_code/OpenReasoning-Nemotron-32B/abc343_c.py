def main():
	N = int(input().strip())
	ans = 0
	x = 1
	while x <= 1000000:
		cube = x * x * x
		if cube > N:
			break
		s = str(cube)
		if s == s[::-1]:
			ans = cube
		x += 1
	print(ans)

if __name__ == '__main__':
	main()