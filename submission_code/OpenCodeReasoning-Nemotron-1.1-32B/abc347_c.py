import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	
	n = int(data[0])
	A = int(data[1])
	B = int(data[2])
	D = list(map(int, data[3:3+n]))
	T = A + B
	
	intervals = []
	for d in D:
		r = d % T
		if r < A:
			low = A - r
			high = T - r - 1
			if low <= high:
				intervals.append((low, high))
		else:
			low = 0
			high = T - r - 1
			intervals.append((low, high))
	
	if not intervals:
		print("Yes")
		return
		
	intervals.sort(key=lambda x: x[0])
	merged = []
	start, end = intervals[0]
	for i in range(1, len(intervals)):
		a, b = intervals[i]
		if a <= end + 1:
			end = max(end, b)
		else:
			merged.append((start, end))
			start, end = a, b
	merged.append((start, end))
	
	total_covered = 0
	for inter in merged:
		a, b = inter
		total_covered += (b - a + 1)
		if total_covered > T:
			break
			
	if total_covered == T:
		print("No")
	else:
		print("Yes")

if __name__ == "__main__":
	main()