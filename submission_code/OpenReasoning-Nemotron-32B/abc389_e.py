import sys
import heapq

def feasible(T, P, M, S_int, N):
	if T == 0:
		return True
		
	base = [0] * N
	T0 = 0
	for i in range(N):
		denom = P[i] * S_int
		base_i_val = (T * 10**18) // denom
		if base_i_val > T:
			base_i_val = T
		base[i] = base_i_val
		T0 += base_i_val

	if T0 > T:
		heap = []
		for i in range(N):
			if base[i] > 0:
				saving = (2 * base[i] - 1) * P[i]
				heapq.heappush(heap, (-saving, i))
		to_remove = T0 - T
		while to_remove > 0 and heap:
			neg_saving, i = heapq.heappop(heap)
			base[i] -= 1
			to_remove -= 1
			if base[i] > 0:
				new_saving = (2 * base[i] - 1) * P[i]
				heapq.heappush(heap, (-new_saving, i))
		T0 = T0 - (T0 - T - to_remove)

	cost0 = 0
	for i in range(N):
		cost0 += base[i] * base[i] * P[i]
	if cost0 > M:
		return False

	to_add = T - T0
	if to_add == 0:
		return cost0 <= M

	heap = []
	for i in range(N):
		mc = (2 * base[i] + 1) * P[i]
		heapq.heappush(heap, (mc, i))
	
	total_cost = cost0
	for _ in range(to_add):
		mc, i = heapq.heappop(heap)
		total_cost += mc
		if total_cost > M:
			return False
		base[i] += 1
		new_mc = (2 * base[i] + 1) * P[i]
		heapq.heappush(heap, (new_mc, i))
		
	return total_cost <= M

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	N = int(data[0])
	M = int(data[1])
	P = list(map(int, data[2:2+N]))
	
	if M == 0:
		print(0)
		return
		
	S_float = 0.0
	for p in P:
		S_float += 1.0 / p
		
	if S_float == 0:
		print(0)
		return
		
	S_int = round(S_float * 10**18)
	
	if M < min(P):
		print(0)
		return
		
	T_high = int((M * S_float) ** 0.5) + N + 100
	T_high = min(T_high, 10**18)
	
	low, high = 0, T_high
	ans = 0
	while low <= high:
		mid = (low + high) // 2
		if feasible(mid, P, M, S_int, N):
			ans = mid
			low = mid + 1
		else:
			high = mid - 1
			
	print(ans)

if __name__ == "__main__":
	main()