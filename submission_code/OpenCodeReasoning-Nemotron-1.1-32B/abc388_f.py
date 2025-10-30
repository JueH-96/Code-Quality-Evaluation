import bisect
from collections import deque
import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print("No")
		return
		
	it = iter(data)
	N = int(next(it)); M = int(next(it)); A = int(next(it)); B = int(next(it))
	bad_intervals = []
	for _ in range(M):
		L_i = int(next(it)); R_i = int(next(it))
		bad_intervals.append((L_i, R_i))
		
	safe_intervals = []
	if M == 0:
		safe_intervals = [(1, N)]
	else:
		safe_intervals.append((1, bad_intervals[0][0]-1))
		for i in range(M-1):
			safe_intervals.append((bad_intervals[i][1]+1, bad_intervals[i+1][0]-1))
		safe_intervals.append((bad_intervals[-1][1]+1, N))
	
	segment_starts = [seg[0] for seg in safe_intervals]
	base = B + 1
	
	reachable = [set() for _ in range(len(safe_intervals))]
	start0 = safe_intervals[0][0]
	residue0 = start0 % base
	reachable[0].add(residue0)
	
	if N == 1:
		print("Yes")
		return
		
	for i in range(len(safe_intervals)):
		if not reachable[i]:
			continue
		queue = deque(reachable[i])
		visited = set(reachable[i])
		while queue:
			r = queue.popleft()
			for step in range(A, B+1):
				new_r = (r + step) % base
				l_i, r_i = safe_intervals[i]
				p_min = l_i + (r - (l_i % base) + base) % base
				if p_min > r_i:
					continue
					
				q0 = p_min + step
				if q0 > N:
					continue
					
				if (N - q0) % base == 0:
					print("Yes")
					return
					
				j0 = bisect.bisect_left(segment_starts, q0)
				for j in range(j0, len(safe_intervals)):
					seg = safe_intervals[j]
					if seg[0] > N:
						break
					if seg[0] > q0:
						k_min = (seg[0] - q0 + base - 1) // base
					else:
						k_min = 0
					high_bound = min(seg[1], N)
					k_max = (high_bound - q0) // base
					if k_min <= k_max:
						if new_r not in reachable[j]:
							reachable[j].add(new_r)
							if j not in visited:
								visited.add(j)
								
	print("No")

if __name__ == '__main__':
	main()