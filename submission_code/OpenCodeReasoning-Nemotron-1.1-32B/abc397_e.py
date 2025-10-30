import sys
sys.setrecursionlimit(300000)

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	k = int(data[1])
	total_vertices = n * k
	graph = [[] for _ in range(total_vertices + 1)]
	index = 2
	for _ in range(total_vertices - 1):
		u = int(data[index])
		v = int(data[index + 1])
		index += 2
		graph[u].append(v)
		graph[v].append(u)
	
	def dfs(u, parent):
		total_paths = 0
		open_children = []
		
		for v in graph[u]:
			if v == parent:
				continue
			child_dp, child_count = dfs(v, u)
			if child_count < 0:
				return (0, -1)
			total_paths += child_count
			if child_dp == k:
				total_paths += 1
			elif child_dp > 0:
				open_children.append(child_dp)
		
		if len(open_children) == 0:
			dp_u = 1
		elif len(open_children) == 1:
			x = open_children[0]
			if x == k - 1:
				total_paths += 1
				dp_u = 0
			else:
				dp_u = x + 1
		elif len(open_children) == 2:
			if open_children[0] + open_children[1] == k - 1:
				total_paths += 1
				dp_u = 0
			else:
				return (0, -1)
		else:
			return (0, -1)
		
		return (dp_u, total_paths)
	
	root_dp, total_paths = dfs(1, -1)
	
	if total_paths < 0:
		print("No")
	else:
		if root_dp > 0:
			if root_dp == k:
				total_paths += 1
			else:
				print("No")
				return
		if total_paths == n:
			print("Yes")
		else:
			print("No")

if __name__ == "__main__":
	main()