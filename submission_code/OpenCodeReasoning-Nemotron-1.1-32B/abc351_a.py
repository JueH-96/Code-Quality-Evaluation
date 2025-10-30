def main():
	takahashi = list(map(int, input().split()))
	aoki = list(map(int, input().split()))
	
	T = sum(takahashi)
	A = sum(aoki)
	
	x = T - A + 1
	print(x)

if __name__ == '__main__':
	main()