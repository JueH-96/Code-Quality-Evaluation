def main():
	n, m = map(int, input().split())
	total = 0
	power = 1
	for i in range(m + 1):
		total += power
		if total > 10**9:
			print("inf")
			return
		if i < m:
			power *= n
	print(total)

if __name__ == "__main__":
	main()