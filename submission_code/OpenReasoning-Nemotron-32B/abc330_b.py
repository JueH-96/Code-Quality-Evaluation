import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	L = int(data[1])
	R = int(data[2])
	A = list(map(int, data[3:3+n]))
	
	res = []
	for a in A:
		if a < L:
			res.append(str(L))
		elif a > R:
			res.append(str(R))
		else:
			res.append(str(a))
			
	print(" ".join(res))

if __name__ == "__main__":
	main()