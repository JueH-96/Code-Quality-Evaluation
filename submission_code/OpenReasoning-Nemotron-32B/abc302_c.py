import sys
import itertools

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print("No")
		return
	
	n, m = map(int, data[0].split())
	strings = [line.strip() for line in data[1:1+n]]
	
	for perm in itertools.permutations(strings):
		valid = True
		for i in range(len(perm) - 1):
			s1 = perm[i]
			s2 = perm[i+1]
			count = 0
			for j in range(m):
				if s1[j] != s2[j]:
					count += 1
					if count > 1:
						break
			if count != 1:
				valid = False
				break
		if valid:
			print("Yes")
			return
			
	print("No")

if __name__ == "__main__":
	main()