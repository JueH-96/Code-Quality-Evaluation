import sys
from collections import defaultdict, deque
import bisect

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	output_lines = []
	for _ in range(t):
		n = int(data[index]); index += 1
		k = int(data[index]); index += 1
		A = list(map(int, data[index:index+n])); index += n
		B = list(map(int, data[index:index+n])); index += n

		if A == B:
			output_lines.append("Yes")
			continue

		if set(B) - set(A):
			output_lines.append("No")
			continue

		current = A[:]
		lists = defaultdict(list)
		sets = defaultdict(set)
		for j in range(n):
			if current[j] != B[j]:
				x = B[j]
				lists[x].append(j)
				sets[x].add(j)
				
		for x in lists:
			lists[x].sort()
		
		queue = deque(range(n))
		
		while queue:
			i = queue.popleft()
			val = current[i]
			
			if val not in sets or not sets[val]:
				continue
				
			low_bound = i - k
			high_bound = i + k
			list_val = lists[val]
			set_val = sets[val]
			
			left_idx = bisect.bisect_left(list_val, low_bound)
			right_idx = bisect.bisect_right(list_val, high_bound) - 1
			
			if left_idx > right_idx:
				continue
				
			for pos in range(left_idx, right_idx + 1):
				j = list_val[pos]
				if j in set_val:
					current[j] = val
					set_val.remove(j)
					queue.append(j)
					
		if current == B:
			output_lines.append("Yes")
		else:
			output_lines.append("No")
			
	print("
".join(output_lines))

if __name__ == '__main__':
	main()