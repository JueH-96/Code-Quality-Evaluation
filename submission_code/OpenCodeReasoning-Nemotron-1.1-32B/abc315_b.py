def main():
	M = int(input().strip())
	D_list = list(map(int, input().split()))
	
	total_days = sum(D_list)
	middle_day = (total_days + 1) // 2
	
	total_so_far = 0
	for i in range(M):
		total_so_far += D_list[i]
		if total_so_far >= middle_day:
			month = i + 1
			day = middle_day - (total_so_far - D_list[i])
			print(f"{month} {day}")
			return

if __name__ == "__main__":
	main()