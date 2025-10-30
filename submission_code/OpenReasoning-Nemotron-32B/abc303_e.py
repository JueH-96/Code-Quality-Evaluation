import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	deg = [0] * (n + 1)
	
	index = 1
	for _ in range(n - 1):
		u = int(data[index])
		v = int(data[index + 1])
		index += 2
		deg[u] += 1
		deg[v] += 1
		
	n1 = 0
	n3p = 0
	for i in range(1, n + 1):
		if deg[i] == 1:
			n1 += 1
		elif deg[i] >= 3:
			n3p += 1
			
	M = (n - n1 + 2) // 3
	center_deg2_count = M - n3p
	
	levels = []
	for i in range(1, n + 1):
		if deg[i] >= 3:
			levels.append(deg[i])
	levels.extend([2] * center_deg2_count)
	levels.sort()
	
	print(" ".join(map(str, levels)))

if __name__ == "__main__":
	main()