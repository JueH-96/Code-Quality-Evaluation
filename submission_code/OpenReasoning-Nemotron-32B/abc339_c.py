def main():
	n = int(input().strip())
	a_list = list(map(int, input().split()))
	
	total = 0
	min_cum = 0
	for a in a_list:
		total += a
		if total < min_cum:
			min_cum = total
	
	X0 = max(0, -min_cum)
	result = X0 + total
	print(result)

if __name__ == "__main__":
	main()