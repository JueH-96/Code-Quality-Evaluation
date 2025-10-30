def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	C = []
	for a in A:
		C.append((a, 0))
	for b in B:
		C.append((b, 1))
		
	C.sort(key=lambda x: x[0])
	
	found = False
	for i in range(len(C) - 1):
		if C[i][1] == 0 and C[i+1][1] == 0:
			found = True
			break
			
	print("Yes" if found else "No")

if __name__ == "__main__":
	main()