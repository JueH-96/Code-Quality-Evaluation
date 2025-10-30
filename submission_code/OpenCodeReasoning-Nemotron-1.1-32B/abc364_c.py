import sys
import heapq

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	X = int(data[1])
	Y = int(data[2])
	A = list(map(int, data[3:3+n]))
	B = list(map(int, data[3+n:3+2*n]))
	
	for i in range(n):
		if A[i] > X or B[i] > Y:
			print(1)
			return
			
	total_A = sum(A)
	total_B = sum(B)
	if total_A <= X and total_B <= Y:
		print(n)
		return
		
	dishes = list(zip(A, B))
	dishes.sort(key=lambda x: x[0])
	
	def can(k):
		vec = []
		for a, b in dishes:
			if a <= X and b <= Y:
				vec.append((a, b))
		if len(vec) < k:
			return False
			
		pq = []
		sumA = 0
		sumB = 0
		for i in range(k-1):
			a_val, b_val = vec[i]
			sumA += a_val
			sumB += b_val
			heapq.heappush(pq, -b_val)
			
		for i in range(k-1, len(vec)):
			a_val, b_val = vec[i]
			while pq and -pq[0] > b_val:
				largest_neg = heapq.heappop(pq)
				largest = -largest_neg
				sumB -= largest
				sumB += b_val
				heapq.heappush(pq, -b_val)
			if sumA + a_val > X or sumB + b_val > Y:
				return True
			sumA += a_val
			sumB += b_val
			heapq.heappush(pq, -b_val)
			largest_neg = heapq.heappop(pq)
			largest = -largest_neg
			sumB -= largest
		return False
		
	low = 2
	high = n
	ans = n
	while low <= high:
		mid = (low + high) // 2
		if can(mid):
			ans = mid
			high = mid - 1
		else:
			low = mid + 1
	print(ans)

if __name__ == '__main__':
	main()