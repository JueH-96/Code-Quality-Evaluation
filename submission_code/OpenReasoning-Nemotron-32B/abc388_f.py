import heapq
import sys

INF = 10**18

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	N = int(data[0]); M = int(data[1]); A = int(data[2]); B = int(data[3])
	bad_intervals = []
	index = 4
	for i in range(M):
		L = int(data[index]); R = int(data[index+1]); index += 2
		bad_intervals.append((L, R))
	
	safe_intervals = []
	current = 1
	for (L, R) in bad_intervals:
		if current < L:
			safe_intervals.append((current, L-1))
		current = R + 1
	if current <= N:
		safe_intervals.append((current, N))
	
	base = B
	
	if not safe_intervals:
		print("No")
		return
		
	s0, e0 = safe_intervals[0]
	if 1 < s0 or 1 > e0:
		print("No")
		return
		
	dist = [INF] * base
	start_res = 1 % base
	dist[start_res] = 1

	def bfs_interval(s, e, dist, A, B, base):
		heap = []
		for r in range(base):
			if dist[r] <= e:
				heapq.heappush(heap, (dist[r], r))
		while heap:
			pos, r = heapq.heappop(heap)
			if pos != dist[r]:
				continue
			for i in range(A, B+1):
				new_pos = pos + i
				if new_pos > e:
					continue
				new_r = (r + i) % base
				if new_pos < dist[new_r]:
					dist[new_r] = new_pos
					heapq.heappush(heap, (new_pos, new_r))
		return dist

	dist = bfs_interval(s0, e0, dist, A, B, base)
	
	if len(safe_intervals) == 1:
		r0 = N % base
		if dist[r0] <= N:
			print("Yes")
		else:
			print("No")
		return

	for j in range(len(safe_intervals)-1):
		s_cur, e_cur = safe_intervals[j]
		s_next, e_next = safe_intervals[j+1]
		
		L_jump = max(s_cur, s_next - B)
		R_jump = min(e_cur, e_next - A)
		
		if L_jump > R_jump:
			print("No")
			return
			
		dist_next = [INF] * base
		
		for r in range(base):
			if dist[r] > R_jump:
				continue
			L0 = max(L_jump, dist[r])
			t = (r - L0) % base
			candidate = L0 + t
			if candidate > R_jump:
				continue
				
			for residue_next in range(base):
				d = (residue_next - r) % base
				i0 = None
				for i_val in range(A, B+1):
					if i_val % base == d:
						i0 = i_val
						break
				if i0 is None:
					continue
				landing_pos = candidate + i0
				if landing_pos < s_next or landing_pos > e_next:
					continue
				if landing_pos < dist_next[residue_next]:
					dist_next[residue_next] = landing_pos
					
		if all(d == INF for d in dist_next):
			print("No")
			return
			
		dist = bfs_interval(s_next, e_next, dist_next, A, B, base)
		
	r0 = N % base
	if dist[r0] <= N:
		print("Yes")
	else:
		print("No")

if __name__ == '__main__':
	main()