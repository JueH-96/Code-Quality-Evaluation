import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	
	n = int(data[0])
	bases = []
	index = 1
	for i in range(n):
		w = int(data[index])
		x = int(data[index + 1])
		index += 2
		bases.append((w, x))
	
	critical_points = set()
	critical_points.add(0.0)
	critical_points.add(24.0)
	
	for w, x in bases:
		a1 = max(0, 9 - x)
		b1 = min(24, 17 - x)
		if a1 < b1:
			critical_points.add(a1)
			critical_points.add(b1)
		
		a2 = max(0, 33 - x)
		b2 = min(24, 41 - x)
		if a2 < b2:
			critical_points.add(a2)
			critical_points.add(b2)
	
	sorted_critical = sorted(critical_points)
	candidate_T = set()
	for t in sorted_critical:
		if t < 24:
			candidate_T.add(t)
	
	for i in range(len(sorted_critical) - 1):
		a = sorted_critical[i]
		b = sorted_critical[i + 1]
		if a < 24 and b > a:
			mid = (a + b) / 2.0
			if mid < 24:
				candidate_T.add(mid)
	
	max_total = 0
	for T in candidate_T:
		total = 0
		for w, x in bases:
			t_local = (x + T) % 24
			if 9 <= t_local <= 17:
				total += w
		if total > max_total:
			max_total = total
	
	print(max_total)

if __name__ == "__main__":
	main()