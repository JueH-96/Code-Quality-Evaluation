import sys

def main():
	data = sys.stdin.read().split()
	n = int(data[0])
	arr = list(map(int, data[1:1+n]))
	
	total_intervals = n * (n + 1) // 2
	pos = [[] for _ in range(n + 1)]
	
	for i in range(n):
		x = arr[i]
		pos[x].append(i)
	
	T0_arr = [0] * (n + 1)
	
	for x in range(1, n + 1):
		if not pos[x]:
			T0_arr[x] = total_intervals
		else:
			T0 = 0
			first_occ = pos[x][0]
			T0 += first_occ * (first_occ + 1) // 2
			
			for j in range(1, len(pos[x])):
				gap_len = pos[x][j] - pos[x][j - 1] - 1
				T0 += gap_len * (gap_len + 1) // 2
				
			last_occ = pos[x][-1]
			right_gap = n - 1 - last_occ
			T0 += right_gap * (right_gap + 1) // 2
			
			T0_arr[x] = T0
			
	S1 = 0
	for x in range(1, n + 1):
		S1 += (total_intervals - T0_arr[x])
		
	S2 = 0
	for x in range(1, n):
		if T0_arr[x] == total_intervals or T0_arr[x + 1] == total_intervals:
			continue
			
		list1 = pos[x]
		list2 = pos[x + 1]
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
			
		T_neither = 0
		first_merged = merged[0]
		T_neither += first_merged * (first_merged + 1) // 2
		
		for j in range(1, len(merged)):
			gap_len = merged[j] - merged[j - 1] - 1
			T_neither += gap_len * (gap_len + 1) // 2
			
		last_merged = merged[-1]
		right_gap = n - 1 - last_merged
		T_neither += right_gap * (right_gap + 1) // 2
		
		count_both = total_intervals - T0_arr[x] - T0_arr[x + 1] + T_neither
		S2 += count_both
		
	ans = S1 - S2
	print(ans)

if __name__ == "__main__":
	main()