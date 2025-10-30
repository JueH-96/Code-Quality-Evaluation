import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(-1)
		return
	
	n = int(data[0])
	x = int(data[1])
	A = list(map(int, data[2:2+n]))
	
	if n < 3:
		print(-1)
		return
		
	B = [(A[i], i) for i in range(n)]
	B.sort(key=lambda x: x[0])
	
	found = False
	for i in range(n-2):
		a_val = B[i][0]
		if 3 * a_val > x:
			break
			
		if i + 2 < n:
			if a_val + B[i+1][0] + B[i+2][0] > x:
				break
				
		if a_val + B[n-1][0] + B[n-2][0] < x:
			continue
			
		j = i + 1
		k = n - 1
		while j < k:
			total = a_val + B[j][0] + B[k][0]
			if total == x:
				idx_i = B[i][1]
				idx_j = B[j][1]
				idx_k = B[k][1]
				indices = sorted([idx_i, idx_j, idx_k])
				print(f"{indices[0]+1} {indices[1]+1} {indices[2]+1}")
				found = True
				break
			elif total < x:
				j += 1
			else:
				k -= 1
				
		if found:
			break
			
	if not found:
		print(-1)

if __name__ == '__main__':
	main()