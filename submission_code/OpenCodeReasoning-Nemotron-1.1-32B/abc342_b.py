def main():
	n = int(input().strip())
	arr = list(map(int, input().split()))
	
	pos_dict = {}
	for idx, person in enumerate(arr):
		pos_dict[person] = idx
		
	q = int(input().strip())
	for _ in range(q):
		a, b = map(int, input().split())
		if pos_dict[a] < pos_dict[b]:
			print(a)
		else:
			print(b)

if __name__ == '__main__':
	main()