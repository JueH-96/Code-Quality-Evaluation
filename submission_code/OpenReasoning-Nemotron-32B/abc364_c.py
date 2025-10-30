import heapq
import sys

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
			
	max_A = max(A)
	max_B = max(B)
	index_A = A.index(max_A)
	second_max_A = -1
	for i in range(n):
		if i == index_A:
			continue
		if A[i] > second_max_A:
			second_max_A = A[i]
	second_max_B = -1
	index_B = B.index(max_B)
	for i in range(n):
		if i == index_B:
			continue
		if B[i] > second_max_B:
			second_max_B = B[i]
			
	if max_A + second_max_A > X or max_B + second_max_B > Y:
		print(2)
		return
		
	total_A = sum(A)
	total_B = sum(B)
	if total_A <= X and total_B <= Y:
		print(n)
		return
		
	ans = 10**18
	for i in range(n):
		other_dishes = []
		for j in range(n):
			if j == i:
				continue
			other_dishes.append((A[j], B[j]))
			
		other_dishes.sort(key=lambda x: x[0])
		heap = []
		sumA = 0
		sumB = 0
		count = 0
		found = False
		for j in range(len(other_dishes)):
			a_val, b_val = other_dishes[j]
			sumA += a_val
			sumB += b_val
			count += 1
			heapq.heappush(heap, (-b_val, a_val))
			
			while (sumA > X or sumB > Y) and heap:
				neg_b, a_val_pop = heapq.heappop(heap)
				b_val_pop = -neg_b
				sumA -= a_val_pop
				sumB -= b_val_pop
				count -= 1
				
			if sumA > X - A[i] or sumB > Y - B[i]:
				s_i = count
				found = True
				break
				
		if found:
			candidate = s_i + 1
			if candidate < ans:
				ans = candidate
		else:
			pass
			
	if ans == 10**18:
		ans = n
		
	print(ans)

if __name__ == "__main__":
	main()