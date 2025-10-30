import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	
	n = int(data[0])
	m = int(data[1])
	products = []
	index = 2
	for i in range(n):
		p = int(data[index])
		index += 1
		c = int(data[index])
		index += 1
		funcs = list(map(int, data[index:index+c]))
		index += c
		products.append((p, set(funcs)))
	
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			p_i, set_i = products[i]
			p_j, set_j = products[j]
			if p_i < p_j:
				continue
			if not set_i.issubset(set_j):
				continue
			if p_i > p_j or len(set_j) > len(set_i):
				print("Yes")
				return
				
	print("No")

if __name__ == '__main__':
	main()