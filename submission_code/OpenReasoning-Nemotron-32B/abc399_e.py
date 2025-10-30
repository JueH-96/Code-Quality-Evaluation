import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		return
	n = int(data[0].strip())
	s = data[1].strip()
	t = data[2].strip()
	
	if s == t:
		print(0)
		return
		
	mapping_dict = {}
	for i in range(n):
		if s[i] != t[i]:
			if s[i] in mapping_dict:
				if mapping_dict[s[i]] != t[i]:
					print(-1)
					return
			else:
				mapping_dict[s[i]] = t[i]
				
	V = set()
	for char in mapping_dict:
		if mapping_dict[char] != char:
			V.add(char)
			
	if len(V) == 0:
		print(0)
		return
		
	visited = set()
	cycles = 0
	
	for char in V:
		if char in visited:
			continue
		path = {}
		current = char
		while current in V and current not in visited:
			visited.add(current)
			path[current] = len(path)
			current = mapping_dict[current]
			
		if current in path:
			cycles += 1
			
	ans = len(V) + cycles
	print(ans)

if __name__ == '__main__':
	main()