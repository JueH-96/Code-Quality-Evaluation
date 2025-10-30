from functools import cmp_to_key
import sys

def main():
	data = sys.stdin.read().splitlines()
	n = int(data[0])
	people = []
	for i in range(1, n + 1):
		a, b = map(int, data[i].split())
		people.append((i, a, b))
	
	def compare(x, y):
		a1, b1 = x[1], x[2]
		a2, b2 = y[1], y[2]
		prod1 = a1 * b2
		prod2 = a2 * b1
		if prod1 > prod2:
			return -1
		elif prod1 < prod2:
			return 1
		else:
			if x[0] < y[0]:
				return -1
			elif x[0] > y[0]:
				return 1
			else:
				return 0
				
	sorted_people = sorted(people, key=cmp_to_key(compare))
	indices = [str(x[0]) for x in sorted_people]
	print(" ".join(indices))

if __name__ == '__main__':
	main()