import sys

def main():
	data = sys.stdin.read().split()
	it = iter(data)
	n = int(next(it)); m = int(next(it)); sx = int(next(it)); sy = int(next(it))
	
	houses = []
	for i in range(n):
		x = int(next(it)); y = int(next(it))
		houses.append((x, y))
		
	moves = []
	for i in range(m):
		d = next(it); c = int(next(it))
		moves.append((d, c))
		
	x, y = sx, sy
	segments = []
	for d, c in moves:
		if d == 'U':
			next_y = y + c
			seg = ('V', x, min(y, next_y), max(y, next_y))
			y = next_y
		elif d == 'D':
			next_y = y - c
			seg = ('V', x, min(y, next_y), max(y, next_y))
			y = next_y
		elif d == 'R':
			next_x = x + c
			seg = ('H', min(x, next_x), max(x, next_x), y)
			x = next_x
		elif d == 'L':
			next_x = x - c
			seg = ('H', min(x, next_x), max(x, next_x), y)
			x = next_x
		segments.append(seg)
		
	horizontal_intervals = {}
	vertical_intervals = {}
	
	for seg in segments:
		if seg[0] == 'H':
			_, min_x, max_x, y_val = seg
			if y_val not in horizontal_intervals:
				horizontal_intervals[y_val] = []
			horizontal_intervals[y_val].append((min_x, max_x))
		else:
			_, x_val, min_y, max_y = seg
			if x_val not in vertical_intervals:
				vertical_intervals[x_val] = []
			vertical_intervals[x_val].append((min_y, max_y))
			
	for y_val, intervals in horizontal_intervals.items():
		intervals.sort(key=lambda inter: inter[0])
		merged = []
		if not intervals:
			continue
		start, end = intervals[0]
		for i in range(1, len(intervals)):
			if intervals[i][0] <= end + 1:
				end = max(end, intervals[i][1])
			else:
				merged.append((start, end))
				start, end = intervals[i]
		merged.append((start, end))
		horizontal_intervals[y_val] = merged
		
	for x_val, intervals in vertical_intervals.items():
		intervals.sort(key=lambda inter: inter[0])
		merged = []
		if not intervals:
			continue
		start, end = intervals[0]
		for i in range(1, len(intervals)):
			if intervals[i][0] <= end + 1:
				end = max(end, intervals[i][1])
			else:
				merged.append((start, end))
				start, end = intervals[i]
		merged.append((start, end))
		vertical_intervals[x_val] = merged
		
	count = 0
	for house in houses:
		a, b = house
		found = False
		if b in horizontal_intervals:
			intervals_list = horizontal_intervals[b]
			low, high = 0, len(intervals_list) - 1
			while low <= high:
				mid = (low + high) // 2
				l, r = intervals_list[mid]
				if a < l:
					high = mid - 1
				elif a > r:
					low = mid + 1
				else:
					found = True
					break
			if found:
				count += 1
				continue
				
		if a in vertical_intervals:
			intervals_list = vertical_intervals[a]
			low, high = 0, len(intervals_list) - 1
			while low <= high:
				mid = (low + high) // 2
				l, r = intervals_list[mid]
				if b < l:
					high = mid - 1
				elif b > r:
					low = mid + 1
				else:
					found = True
					break
			if found:
				count += 1
				
	print(f"{x} {y} {count}")

if __name__ == "__main__":
	main()