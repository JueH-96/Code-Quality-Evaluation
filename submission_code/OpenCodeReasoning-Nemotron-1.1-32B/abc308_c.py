import sys
from fractions import Fraction

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	people = []
	index = 1
	for i in range(n):
		a = int(data[index])
		b = int(data[index + 1])
		index += 2
		total = a + b
		tail_rate = Fraction(b, total)
		people.append((tail_rate, i + 1))
	
	people.sort(key=lambda x: (x[0], x[1]))
	
	result = [str(idx) for _, idx in people]
	print(" ".join(result))

if __name__ == '__main__':
	main()