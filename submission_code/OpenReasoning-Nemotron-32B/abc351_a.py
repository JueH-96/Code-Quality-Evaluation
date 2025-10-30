def main():
	taka = list(map(int, input().split()))
	aoki = list(map(int, input().split()))
	total_taka = sum(taka)
	total_aoki = sum(aoki)
	ans = total_taka - total_aoki + 1
	print(ans)

if __name__ == "__main__":
	main()