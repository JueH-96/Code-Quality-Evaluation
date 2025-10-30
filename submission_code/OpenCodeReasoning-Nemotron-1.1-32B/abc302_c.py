import itertools

def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		print("No")
		return
	n, m = map(int, data[0].split())
	strings = [line.strip() for line in data[1:1+n]]
	
	dist = [[0] * n for _ in range(n)]
	for i in range(n):
		for j in range(n):
			if i == j:
				dist[i][j] = 0
			else:
				count = 0
				for k in range(m):
					if strings[i][k] != strings[j][k]:
						count += 1
				dist[i][j] = count
				
	for perm in itertools.permutations(range(n)):
		valid = True
		for i in range(n-1):
			if dist[perm[i]][perm[i+1]] != 1:
				valid = False
				break
		if valid:
			print("Yes")
			return
			
	print("No")

if __name__ == "__main__":
	main()