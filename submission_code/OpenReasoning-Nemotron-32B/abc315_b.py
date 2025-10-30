def main():
	M = int(input().strip())
	D_list = list(map(int, input().split()))
	
	total_days = sum(D_list)
	mid = (total_days + 1) // 2
	
	prefix = 0
	for i in range(M):
		if prefix + D_list[i] >= mid:
			day = mid - prefix
			month = i + 1
			print(f"{month} {day}")
			return
		prefix += D_list[i]

if __name__ == "__main__":
	main()