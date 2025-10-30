import sys

def main():
	data = sys.stdin.read().split()
	if not data:
		print(0)
		return
	n = int(data[0])
	X = int(data[1])
	processes = []
	index = 2
	for i in range(n):
		A = int(data[index])
		P = int(data[index+1])
		B = int(data[index+2])
		Q = int(data[index+3])
		index += 4
		processes.append((A, P, B, Q))
	
	low = 0
	high = 10**18
	ans = 0
	while low <= high:
		mid = (low + high) // 2
		total_cost = 0
		valid = True
		for (A, P, B, Q) in processes:
			if mid == 0:
				cost_i = 0
			else:
				cost1 = 10**18
				for k in range(0, B):
					remaining = mid - k * A
					if remaining <= 0:
						cost_candidate = k * P
					else:
						l_needed = (remaining + B - 1) // B
						cost_candidate = k * P + l_needed * Q
					if cost_candidate < cost1:
						cost1 = cost_candidate
				
				cost2 = 10**18
				for l in range(0, A):
					remaining = mid - l * B
					if remaining <= 0:
						cost_candidate = l * Q
					else:
						k_needed = (remaining + A - 1) // A
						cost_candidate = k_needed * P + l * Q
					if cost_candidate < cost2:
						cost2 = cost_candidate
				
				cost_i = min(cost1, cost2)
			total_cost += cost_i
			if total_cost > X:
				break
		
		if total_cost <= X:
			ans = mid
			low = mid + 1
		else:
			high = mid - 1
	
	print(ans)

if __name__ == "__main__":
	main()