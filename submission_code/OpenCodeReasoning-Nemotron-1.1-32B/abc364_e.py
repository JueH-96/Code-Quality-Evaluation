import sys

def main():
	data = sys.stdin.read().strip().split()
	if not data:
		return
	n = int(data[0])
	X = int(data[1])
	Y = int(data[2])
	A = []
	B = []
	index = 3
	for i in range(n):
		a = int(data[index])
		b = int(data[index+1])
		index += 2
		A.append(a)
		B.append(b)
	
	ans = 0
	
	for i in range(n):
		dp = [[-1] * (Y+1) for _ in range(X+1)]
		dp[0][0] = 0
		states = set()
		states.add((0,0))
		
		for j in range(n):
			if j == i:
				continue
			new_dp = [row[:] for row in dp]
			new_states = set(states)
			for (x, y) in states:
				nx = x + A[j]
				ny = y + B[j]
				if nx <= X and ny <= Y:
					if new_dp[nx][ny] < dp[x][y] + 1:
						new_dp[nx][ny] = dp[x][y] + 1
						new_states.add((nx, ny))
			dp = new_dp
			states = new_states
		
		for (x, y) in states:
			if dp[x][y] >= 0:
				if dp[x][y] + 1 > ans:
					ans = dp[x][y] + 1
					
	dp = [[-1] * (Y+1) for _ in range(X+1)]
	dp[0][0] = 0
	states = set()
	states.add((0,0))
	for i in range(n):
		new_dp = [row[:] for row in dp]
		new_states = set(states)
		for (x, y) in states:
			nx = x + A[i]
			ny = y + B[i]
			if nx <= X and ny <= Y:
				if new_dp[nx][ny] < dp[x][y] + 1:
					new_dp[nx][ny] = dp[x][y] + 1
					new_states.add((nx, ny))
		dp = new_dp
		states = new_states
		
	for (x, y) in states:
		if dp[x][y] >= 0:
			if dp[x][y] > ans:
				ans = dp[x][y]
				
	print(ans)

if __name__ == '__main__':
	main()