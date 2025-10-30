import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	index = 0
	n = int(data[index]); index += 1
	A = list(map(int, data[index:index+n])); index += n
	m = int(data[index]); index += 1
	B = list(map(int, data[index:index+m])); index += m
	l = int(data[index]); index += 1
	C = list(map(int, data[index:index+l])); index += l
	q = int(data[index]); index += 1
	X = list(map(int, data[index:index+q]))
	
	T = set()
	for a in A:
		for b in B:
			for c in C:
				total = a + b + c
				T.add(total)
				
	for x in X:
		if x in T:
			print("Yes")
		else:
			print("No")

if __name__ == '__main__':
	main()