def main():
	data = input().split()
	n = int(data[0])
	l = int(data[1])
	r = int(data[2])
	
	arr = list(range(1, n+1))
	
	segment = arr[l-1:r]
	segment.reverse()
	arr[l-1:r] = segment
	
	print(" ".join(map(str, arr)))

if __name__ == "__main__":
	main()