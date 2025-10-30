import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	visited = [False] * (n + 1)
	
	for i in range(1, n + 1):
		if not visited[i]:
			path_dict = {}
			path_list = []
			cur = i
			while not visited[cur]:
				visited[cur] = True
				path_dict[cur] = len(path_list)
				path_list.append(cur)
				cur = A[cur - 1]
				
			if cur in path_dict:
				idx = path_dict[cur]
				cycle = path_list[idx:]
				print(len(cycle))
				print(' '.join(map(str, cycle)))
				return

if __name__ == "__main__":
	main()