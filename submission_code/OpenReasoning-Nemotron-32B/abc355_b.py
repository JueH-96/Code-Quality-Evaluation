def main():
	import sys
	data = sys.stdin.read().split()
	n = int(data[0])
	m = int(data[1])
	A = list(map(int, data[2:2+n]))
	B = list(map(int, data[2+n:2+n+m]))
	
	setA = set(A)
	combined = sorted(A + B)
	
	found = False
	for i in range(len(combined) - 1):
		if combined[i] in setA and combined[i+1] in setA:
			found = True
			break
			
	print("Yes" if found else "No")

if __name__ == "__main__":
	main()