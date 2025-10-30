import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	q = int(data[1])
	queries = list(map(int, data[2:2+q]))
	
	sizes = [0] * q
	events = [[] for _ in range(n+1)]
	
	current_set = set()
	current_size = 0
	
	for i in range(q):
		x = queries[i]
		if x in current_set:
			current_set.remove(x)
			current_size -= 1
		else:
			current_set.add(x)
			current_size += 1
		sizes[i] = current_size
		events[x].append(i)
	
	prefix = [0] * (q+1)
	for i in range(1, q+1):
		prefix[i] = prefix[i-1] + sizes[i-1]
	
	ans = []
	for j in range(1, n+1):
		total = 0
		ev_list = events[j]
		k = len(ev_list)
		for idx in range(0, k, 2):
			start_index = ev_list[idx]
			if idx + 1 < k:
				end_index = ev_list[idx+1] - 1
			else:
				end_index = q - 1
			total += prefix[end_index+1] - prefix[start_index]
		ans.append(str(total))
	
	print(" ".join(ans))

if __name__ == "__main__":
	main()