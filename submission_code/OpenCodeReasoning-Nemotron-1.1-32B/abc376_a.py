def main():
	n, c = map(int, input().split())
	t = list(map(int, input().split()))
	
	if n == 0:
		print(0)
		return
		
	count = 1
	last_time = t[0]
	
	for i in range(1, n):
		if t[i] - last_time >= c:
			count += 1
			last_time = t[i]
			
	print(count)

if __name__ == "__main__":
	main()