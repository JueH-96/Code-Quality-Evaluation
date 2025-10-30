import sys
import functools

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	n = int(data[0])
	P = []
	index = 1
	for i in range(n):
		x = int(data[index])
		y = int(data[index + 1])
		index += 2
		P.append((x, y, i + 1))
	Q = []
	for i in range(n):
		x = int(data[index])
		y = int(data[index + 1])
		index += 2
		Q.append((x, y, i + 1))
	
	def match(P_list, Q_list):
		n_val = len(P_list)
		if n_val == 0:
			return []
		all_pts = []
		for pt in P_list:
			all_pts.append((pt[0], pt[1], 'P', pt[2]))
		for pt in Q_list:
			all_pts.append((pt[0], pt[1], 'Q', pt[2]))
		
		A = min(all_pts, key=lambda p: (p[0], p[1]))
		
		other = [pt for pt in all_pts if pt != A]
		
		def cmp_angle(B, C):
			bx = B[0] - A[0]
			by = B[1] - A[1]
			cx = C[0] - A[0]
			cy = C[1] - A[1]
			cross = bx * cy - by * cx
			if cross > 0:
				return -1
			elif cross < 0:
				return 1
			else:
				distB = bx * bx + by * by
				distC = cx * cx + cy * cy
				if distB < distC:
					return -1
				elif distB > distC:
					return 1
				else:
					return 0
					
		other_sorted = sorted(other, key=functools.cmp_to_key(cmp_angle))
		
		if A[2] == 'P':
			counter = 0
			chosen = None
			for i, pt in enumerate(other_sorted):
				if pt[2] == 'P':
					counter += 1
				else:
					counter -= 1
				if pt[2] == 'Q' and counter == -1:
					chosen = pt
					break
			if chosen is None:
				return None
			set1 = other_sorted[:i]
			set2 = other_sorted[i + 1:]
			
			set1_P = [(x, y, idx) for (x, y, t, idx) in set1 if t == 'P']
			set1_Q = [(x, y, idx) for (x, y, t, idx) in set1 if t == 'Q']
			set2_P = [(x, y, idx) for (x, y, t, idx) in set2 if t == 'P']
			set2_Q = [(x, y, idx) for (x, y, t, idx) in set2 if t == 'Q']
			
			res1 = match(set1_P, set1_Q)
			res2 = match(set2_P, set2_Q)
			if res1 is None or res2 is None:
				return None
			current = [(A[3], chosen[3])]
			return current + res1 + res2
			
		else:
			counter = 0
			chosen = None
			for i, pt in enumerate(other_sorted):
				if pt[2] == 'Q':
					counter += 1
				else:
					counter -= 1
				if pt[2] == 'P' and counter == -1:
					chosen = pt
					break
			if chosen is None:
				return None
			set1 = other_sorted[:i]
			set2 = other_sorted[i + 1:]
			
			set1_P = [(x, y, idx) for (x, y, t, idx) in set1 if t == 'P']
			set1_Q = [(x, y, idx) for (x, y, t, idx) in set1 if t == 'Q']
			set2_P = [(x, y, idx) for (x, y, t, idx) in set2 if t == 'P']
			set2_Q = [(x, y, idx) for (x, y, t, idx) in set2 if t == 'Q']
			
			res1 = match(set1_P, set1_Q)
			res2 = match(set2_P, set2_Q)
			if res1 is None or res2 is None:
				return None
			current = [(chosen[3], A[3])]
			return current + res1 + res2
			
	matches = match(P, Q)
	if matches is None:
		print(-1)
	else:
		R = [0] * (n + 1)
		for (p_idx, q_idx) in matches:
			R[p_idx] = q_idx
		ans = []
		for i in range(1, n + 1):
			ans.append(str(R[i]))
		print(" ".join(ans))

if __name__ == "__main__":
	main()