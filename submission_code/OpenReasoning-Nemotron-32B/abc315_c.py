import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n = int(data[0])
	groups = {}
	index = 1
	for i in range(n):
		line = data[index].split()
		index += 1
		if not line:
			continue
		f = int(line[0])
		s = int(line[1])
		if f not in groups:
			groups[f] = []
		groups[f].append(s)
	
	candidate_same = 0
	max_vals = []
	
	for flavor, arr in groups.items():
		max1 = arr[0]
		max2 = -10**18
		for j in range(1, len(arr)):
			if arr[j] > max1:
				max2 = max1
				max1 = arr[j]
			elif arr[j] > max2:
				max2 = arr[j]
		if len(arr) >= 2:
			satisfaction = max1 + (max2 // 2)
			if satisfaction > candidate_same:
				candidate_same = satisfaction
		max_vals.append(max1)
	
	if len(max_vals) < 2:
		print(candidate_same)
	else:
		max_vals.sort(reverse=True)
		candidate_diff = max_vals[0] + max_vals[1]
		ans = max(candidate_same, candidate_diff)
		print(ans)

if __name__ == '__main__':
	main()