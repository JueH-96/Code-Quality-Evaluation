import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	N = int(data[0].strip())
	S = data[1].strip()
	T = data[2].strip()
	
	mapping_req = {}
	for i in range(N):
		s_char = S[i]
		t_char = T[i]
		if s_char != t_char:
			if s_char in mapping_req:
				if mapping_req[s_char] != t_char:
					print(-1)
					return
			else:
				mapping_req[s_char] = t_char
				
	num_edges = len(mapping_req)
	if num_edges == 0:
		print(0)
		return
		
	all_nodes = set(mapping_req.keys()) | set(mapping_req.values())
	total_nodes = len(all_nodes)
	
	visited = set()
	cycles = 0
	for node in mapping_req:
		if node not in visited:
			path = []
			cur = node
			while cur is not None and cur not in visited:
				if cur in path:
					idx = path.index(cur)
					cycle_length = len(path) - idx
					if cycle_length == 26:
						print(-1)
						return
					cycles += 1
					break
				path.append(cur)
				if cur in mapping_req:
					cur = mapping_req[cur]
				else:
					cur = None
					
			visited.update(path)
			
	ans = num_edges + cycles
	print(ans)

if __name__ == '__main__':
	main()