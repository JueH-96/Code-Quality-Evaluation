import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
	
	n = int(data[0])
	m = int(data[1])
	index = 2
	products = []
	for i in range(n):
		p = int(data[index]); index += 1
		c = int(data[index]); index += 1
		functions = list(map(int, data[index:index+c]))
		index += c
		products.append((p, set(functions)))
	
	for i in range(n):
		for j in range(n):
			if i == j:
				continue
			p_i, funcs_i = products[i]
			p_j, funcs_j = products[j]
			if p_i >= p_j and funcs_i.issubset(funcs_j):
				if p_i > p_j:
					print("Yes")
					return
				elif p_i == p_j:
					if funcs_i != funcs_j:
						print("Yes")
						return
	print("No")

if __name__ == "__main__":
	main()