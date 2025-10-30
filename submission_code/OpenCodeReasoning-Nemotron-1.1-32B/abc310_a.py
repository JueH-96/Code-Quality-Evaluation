def main():
	n, P, Q = map(int, input().split())
	dishes = list(map(int, input().split()))
	min_dish = min(dishes)
	ans = min(P, Q + min_dish)
	print(ans)

if __name__ == "__main__":
	main()