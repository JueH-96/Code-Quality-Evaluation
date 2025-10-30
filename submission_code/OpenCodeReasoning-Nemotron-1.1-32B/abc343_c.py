def main():
	N = int(input().strip())
	for x in range(10**6, 0, -1):
		cube = x * x * x
		if cube <= N:
			s = str(cube)
			if s == s[::-1]:
				print(cube)
				break

if __name__ == '__main__':
	main()