import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	A = list(map(int, data[1:1+n]))
	
	total_subarrays = n * (n + 1) // 2
	pos = [[] for _ in range(n + 1)]
	
	for i in range(n):
		a = A[i]
		if 1 <= a <= n:
			pos[a].append(i + 1)
	
	without = [0] * (n + 1)
	for x in range(1, n + 1):
		if not pos[x]:
			without[x] = total_subarrays
		else:
			gaps = []
			gaps.append(pos[x][0] - 1)
			for j in range(1, len(pos[x])):
				gaps.append(pos[x][j] - pos[x][j - 1] - 1)
			gaps.append(n - pos[x][-1])
			total_gap = 0
			for g in gaps:
				total_gap += g * (g + 1) // 2
			without[x] = total_gap
	
	ans = 0
	ans += (total_subarrays - without[1])
	
	for x in range(2, n + 1):
		list1 = pos[x - 1]
		list2 = pos[x]
		merged = []
		i, j = 0, 0
		len1 = len(list1)
		len2 = len(list2)
		while i < len1 and j < len2:
			if list1[i] < list2[j]:
				merged.append(list1[i])
				i += 1
			else:
				merged.append(list2[j])
				j += 1
		while i < len1:
			merged.append(list1[i])
			i += 1
		while j < len2:
			merged.append(list2[j])
			j += 1
		
		if not merged:
			without_both = total_subarrays
		else:
			gaps = []
			gaps.append(merged[0] - 1)
			for idx in range(1, len(merged)):
				gaps.append(merged[idx] - merged[idx - 1] - 1)
			gaps.append(n - merged[-1])
			without_both = 0
			for g in gaps:
				without_both += g * (g + 1) // 2
		
		ans += (without[x - 1] - without_both)
	
	print(ans)

if __name__ == "__main__":
	main()