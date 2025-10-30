import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	it = iter(data)
	n = int(next(it)); m = int(next(it)); S_x = int(next(it)); S_y = int(next(it))
	houses = []
	for i in range(n):
		x = int(next(it)); y = int(next(it))
		houses.append((x, y))
	
	moves = []
	for i in range(m):
		d = next(it)
		c = int(next(it))
		moves.append((d, c))
	
	current_x, current_y = S_x, S_y
	horizontal_intervals = defaultdict(list)
	vertical_intervals = defaultdict(list)
	
	for d, c in moves:
		next_x, next_y = current_x, current_y
		if d == 'U':
			next_y += c
		elif d == 'D':
			next_y -= c
		elif d == 'L':
			next_x -= c
		elif d == 'R':
			next_x += c
			
		if current_y == next_y:
			x_min = min(current_x, next_x)
			x_max = max(current_x, next_x)
			horizontal_intervals[current_y].append((x_min, x_max))
		elif current_x == next_x:
			y_min = min(current_y, next_y)
			y_max = max(current_y, next_y)
			vertical_intervals[current_x].append((y_min, y_max))
			
		current_x, current_y = next_x, next_y
		
	merged_horizontal = {}
	for y, intervals in horizontal_intervals.items():
		if not intervals:
			continue
		intervals.sort(key=lambda x: x[0])
		merged = []
		start, end = intervals[0]
		for i in range(1, len(intervals)):
			s, e = intervals[i]
			if s <= end + 1:
				end = max(end, e)
			else:
				merged.append((start, end))
				start, end = s, e
		merged.append((start, end))
		merged_horizontal[y] = merged
		
	merged_vertical = {}
	for x, intervals in vertical_intervals.items():
		if not intervals:
			continue
		intervals.sort(key=lambda x: x[0])
		merged = []
		start, end = intervals[0]
		for i in range(1, len(intervals)):
			s, e = intervals[i]
			if s <= end + 1:
				end = max(end, e)
			else:
				merged.append((start, end))
				start, end = s, e
		merged.append((start, end))
		merged_vertical[x] = merged
		
	count = 0
	for house in houses:
		a, b = house
		found = False
		if b in merged_horizontal:
			intervals_list = merged_horizontal[b]
			low, high = 0, len(intervals_list) - 1
			while low <= high:
				mid = (low + high) // 2
				s, e = intervals_list[mid]
				if a < s:
					high = mid - 1
				elif a > e:
					low = mid + 1
				else:
					found = True
					break
		if found:
			count += 1
			continue
			
		if a in merged_vertical:
			intervals_list = merged_vertical[a]
			low, high = 0, len(intervals_list) - 1
			while low <= high:
				mid = (low + high) // 2
				s, e = intervals_list[mid]
				if b < s:
					high = mid - 1
				elif b > e:
					low = mid + 1
				else:
					found = True
					break
		if found:
			count += 1
			
	print(f"{current_x} {current_y} {count}")

if __name__ == '__main__':
	main()