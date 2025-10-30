import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	P = list(map(int, data[1:1+n]))
	Q = list(map(int, data[1+n:1+2*n]))
	
	res = [0] * n
	for j in range(n):
		bib_index = Q[j] - 1
		target_bib = Q[P[j] - 1]
		res[bib_index] = target_bib
		
	print(" ".join(map(str, res)))

if __name__ == "__main__":
	main()