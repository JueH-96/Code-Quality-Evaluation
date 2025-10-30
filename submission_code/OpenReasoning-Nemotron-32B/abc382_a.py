def main():
	n, d = map(int, input().split())
	s = input().strip()
	count_at = s.count('@')
	initial_empty = n - count_at
	result = initial_empty + d
	print(result)

if __name__ == '__main__':
	main()