import sys
import bisect

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	q = int(data[1])
	R = list(map(int, data[2:2+n]))
	queries = list(map(int, data[2+n:2+n+q]))
	
	R.sort()
	prefix = [0]
	for num in R:
		prefix.append(prefix[-1] + num)
	
	results = []
	for x in queries:
		idx = bisect.bisect_right(prefix, x)
		results.append(str(idx - 1))
	
	print("
".join(results))

if __name__ == "__main__":
	main()