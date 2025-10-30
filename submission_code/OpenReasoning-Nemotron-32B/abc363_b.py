def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	T = int(data[1])
	p = int(data[2])
	L = list(map(int, data[3:3+n]))
	
	d_list = [max(0, T - x) for x in L]
	d_list.sort()
	print(d_list[p-1])

if __name__ == '__main__':
	main()