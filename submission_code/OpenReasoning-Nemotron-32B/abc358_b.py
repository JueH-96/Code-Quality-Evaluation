import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = int(data[1])
	T = list(map(int, data[2:2+n]))
	
	current_time = 0
	results = []
	for t in T:
		start_time = max(current_time, t)
		finish_time = start_time + A
		results.append(str(finish_time))
		current_time = finish_time
	
	print("
".join(results))

if __name__ == "__main__":
	main()