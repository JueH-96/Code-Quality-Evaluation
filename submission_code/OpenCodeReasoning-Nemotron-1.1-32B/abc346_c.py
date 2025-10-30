def main():
	n, k = map(int, input().split())
	a = list(map(int, input().split()))
	total = k * (k + 1) // 2
	s = set(x for x in a if x <= k)
	subtract = sum(s)
	print(total - subtract)

if __name__ == "__main__":
	main()