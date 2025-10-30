def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	A = int(data[1])
	T = list(map(int, data[2:2+n]))
	
	if n == 0:
		return
	
	first_finish = T[0] + A
	print(first_finish)
	
	prev_finish = first_finish
	for i in range(1, n):
		start_time = max(T[i], prev_finish)
		finish_time = start_time + A
		print(finish_time)
		prev_finish = finish_time

if __name__ == "__main__":
	main()