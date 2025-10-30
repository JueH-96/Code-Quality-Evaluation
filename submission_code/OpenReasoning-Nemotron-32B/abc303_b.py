import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	m = int(data[1])
	photos = []
	index = 2
	for i in range(m):
		arr = list(map(int, data[index:index+n]))
		index += n
		photos.append(arr)
	
	all_pos = []
	for arr in photos:
		pos_arr = [0] * (n + 1)
		for idx, person in enumerate(arr):
			pos_arr[person] = idx
		all_pos.append(pos_arr)
	
	count = 0
	for x in range(1, n + 1):
		for y in range(x + 1, n + 1):
			found_adjacent = False
			for i in range(m):
				p1 = all_pos[i][x]
				p2 = all_pos[i][y]
				if abs(p1 - p2) == 1:
					found_adjacent = True
					break
			if not found_adjacent:
				count += 1
				
	print(count)

if __name__ == "__main__":
	main()