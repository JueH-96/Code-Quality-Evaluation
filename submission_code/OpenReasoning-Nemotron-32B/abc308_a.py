def main():
	data = input().split()
	if len(data) != 8:
		print('No')
		return
	
	try:
		arr = list(map(int, data))
	except:
		print('No')
		return
		
	for x in arr:
		if x < 100 or x > 675 or x % 25 != 0:
			print('No')
			return
			
	for i in range(7):
		if arr[i] > arr[i+1]:
			print('No')
			return
			
	print('Yes')

if __name__ == '__main__':
	main()