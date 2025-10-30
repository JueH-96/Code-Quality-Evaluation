import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	X = list(map(int, data[1:1+n]))
	
	while True:
		X.sort()
		found = False
		for i in range(len(X) - 3):
			if X[i+1] + X[i+2] > X[i] + X[i+3]:
				a = X[i]
				d = X[i+3]
				X[i+1] = a + d - X[i+1]
				X[i+2] = a + d - X[i+2]
				found = True
				break
		if not found:
			break
			
	print(sum(X))

if __name__ == '__main__':
	main()