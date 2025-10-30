import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	intervals = []
	vals = set()
	index = 1
	for i in range(n):
		l = int(data[index])
		r = int(data[index+1])
		index += 2
		intervals.append((l, r))
		vals.add(l)
		vals.add(r)
	
	sorted_vals = sorted(vals)
	comp = {val: idx+1 for idx, val in enumerate(sorted_vals)}
	m = len(sorted_vals)
	
	intervals.sort(key=lambda x: (x[0], x[1]))
	
	fenw = [0] * (m+1)
	
	def fenw_update(i, delta):
		while i <= m:
			fenw[i] += delta
			i += i & -i
			
	def fenw_query(i):
		s = 0
		while i:
			s += fenw[i]
			i -= i & -i
		return s
		
	non_intersect = 0
	for l, r in intervals:
		pos_l = comp[l]
		if pos_l > 1:
			cnt = fenw_query(pos_l-1)
		else:
			cnt = 0
		non_intersect += cnt
		
		pos_r = comp[r]
		fenw_update(pos_r, 1)
		
	total_pairs = n * (n-1) // 2
	ans = total_pairs - non_intersect
	print(ans)

if __name__ == "__main__":
	main()