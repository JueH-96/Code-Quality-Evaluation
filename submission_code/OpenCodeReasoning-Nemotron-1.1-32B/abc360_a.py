def main():
	s = input().strip()
	index_r = s.index('R')
	index_m = s.index('M')
	if index_r < index_m:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()