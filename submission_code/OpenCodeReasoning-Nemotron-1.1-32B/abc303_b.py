def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	photos = []
	index = 2
	for i in range(m):
		arr = list(map(int, data[index:index+n]))
		index += n
		photos.append(arr)
	
	adj_sets = []
	for arr in photos:
		s = set()
		for j in range(len(arr)-1):
			a = arr[j]
			b = arr[j+1]
			if a < b:
				pair = (a, b)
			else:
				pair = (b, a)
			s.add(pair)
		adj_sets.append(s)
	
	bad_count = 0
	for x in range(1, n+1):
		for y in range(x+1, n+1):
			pair = (x, y)
			found = False
			for s in adj_sets:
				if pair in s:
					found = True
					break
			if not found:
				bad_count += 1
				
	print(bad_count)

if __name__ == "__main__":
	main()