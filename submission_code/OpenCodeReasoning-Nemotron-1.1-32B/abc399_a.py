def main():
	n = int(input().strip())
	S = input().strip()
	T = input().strip()
	count = 0
	for i in range(n):
		if S[i] != T[i]:
			count += 1
	print(count)

if __name__ == '__main__':
	main()