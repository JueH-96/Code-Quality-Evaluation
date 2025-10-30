import math
import sys

def main():
	data = sys.stdin.read().strip().split()
	if not data: 
		return
	it = iter(data)
	N = int(next(it)); S = int(next(it)); T = int(next(it))
	segments = []
	for i in range(N):
		A = int(next(it)); B = int(next(it)); C = int(next(it)); D = int(next(it))
		segments.append((A, B, C, D))
	
	lengths = []
	for (A, B, C, D) in segments:
		L = math.sqrt((A - C) ** 2 + (B - D) ** 2)
		lengths.append(L)
	
	dp = [dict() for _ in range(1 << N)]
	dp[0][(0, 0)] = 0.0
	
	for mask in range(1 << N):
		for point, time_val in list(dp[mask].items()):
			for i in range(N):
				if mask & (1 << i):
					continue
				A, B, C, D = segments[i]
				L = lengths[i]
				
				dx1 = point[0] - A
				dy1 = point[1] - B
				d1 = math.sqrt(dx1*dx1 + dy1*dy1)
				total_time1 = time_val + d1 / S + L / T
				new_mask1 = mask | (1 << i)
				new_point1 = (C, D)
				
				if new_point1 in dp[new_mask1]:
					if total_time1 < dp[new_mask1][new_point1]:
						dp[new_mask1][new_point1] = total_time1
				else:
					dp[new_mask1][new_point1] = total_time1
					
				dx2 = point[0] - C
				dy2 = point[1] - D
				d2 = math.sqrt(dx2*dx2 + dy2*dy2)
				total_time2 = time_val + d2 / S + L / T
				new_mask2 = mask | (1 << i)
				new_point2 = (A, B)
				
				if new_point2 in dp[new_mask2]:
					if total_time2 < dp[new_mask2][new_point2]:
						dp[new_mask2][new_point2] = total_time2
				else:
					dp[new_mask2][new_point2] = total_time2
					
	final_mask = (1 << N) - 1
	if not dp[final_mask]:
		print(0.0)
	else:
		ans = min(dp[final_mask].values())
		print(ans)

if __name__ == '__main__':
	main()