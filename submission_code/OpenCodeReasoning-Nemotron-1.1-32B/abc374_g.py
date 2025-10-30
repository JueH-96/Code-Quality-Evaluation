import sys
from collections import defaultdict

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print(0)
		return
	n = int(data[0])
	edges = [line.strip() for line in data[1:1+n]]
	
	if n == 0:
		print(0)
		return
		
	vertices = set()
	for s in edges:
		vertices.add(s[0])
		vertices.add(s[1])
	
	undir_graph = defaultdict(list)
	for s in edges:
		u, v = s[0], s[1]
		undir_graph[u].append(v)
		undir_graph[v].append(u)
	
	visited_undir = set()
	components = []
	for v in vertices:
		if v not in visited_undir:
			comp = []
			stack = [v]
			visited_undir.add(v)
			while stack:
				node = stack.pop()
				comp.append(node)
				for neighbor in undir_graph[node]:
					if neighbor not in visited_undir:
						visited_undir.add(neighbor)
						stack.append(neighbor)
			components.append(comp)
	
	total_walks = 0
	for comp in components:
		subgraph = defaultdict(list)
		rev_subgraph = defaultdict(list)
		for s in edges:
			u, v = s[0], s[1]
			if u in comp and v in comp:
				subgraph[u].append(v)
				rev_subgraph[v].append(u)
				
		visited = set()
		order = []
		
		def dfs1(node):
			visited.add(node)
			for neighbor in subgraph.get(node, []):
				if neighbor not in visited:
					dfs1(neighbor)
			order.append(node)
			
		for node in comp:
			if node not in visited:
				dfs1(node)
				
		visited2 = set()
		scc_list = []
		
		def dfs2(node, scc):
			visited2.add(node)
			scc.append(node)
			for neighbor in rev_subgraph.get(node, []):
				if neighbor not in visited2:
					dfs2(neighbor, scc)
					
		for node in reversed(order):
			if node not in visited2:
				scc = []
				dfs2(node, scc)
				scc_list.append(scc)
				
		scc_index = {}
		for idx, scc in enumerate(scc_list):
			for node in scc:
				scc_index[node] = idx
				
		condensation = defaultdict(set)
		for s in edges:
			u, v = s[0], s[1]
			if u in comp and v in comp:
				i = scc_index[u]
				j = scc_index[v]
				if i != j:
					condensation[i].add(j)
					
		sinks = 0
		for i in range(len(scc_list)):
			if i not in condensation:
				sinks += 1
				
		total_walks += sinks
		
	print(total_walks)

if __name__ == "__main__":
	main()