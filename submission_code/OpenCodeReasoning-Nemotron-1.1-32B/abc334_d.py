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
	
	out_lines = []
	for x in queries:
		pos = bisect.bisect_right(prefix, x)
		out_lines.append(str(pos - 1))
	
	print("
".join(out_lines))

if __name__ == "__main__":
	main()