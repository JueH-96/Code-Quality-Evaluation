import sys
from collections import deque

def main():
	data = sys.stdin.read().split()
	N = int(data[0])
	A = int(data[1])
	X = int(data[2])
	Y = int(data[3])
	
	states = set()
	queue = deque([N])
	states.add(N)
	while queue:
		n = queue.popleft()
		if n == 0:
			continue
		next_states = {n // A}
		for b in range(2, 7):
			next_states.add(n // b)
		for ns in next_states:
			if ns not in states:
				states.add(ns)
				queue.append(ns)
				
	sorted_states = sorted(states)
	dp = {}
	dp[0] = 0.0
	if 1 in states:
		dp[1] = min(X, 6 * Y / 5.0)
	
	for n in sorted_states:
		if n <= 1:
			continue
		op1 = X + dp[n // A]
		total = 0.0
		for b in range(2, 7):
			total += dp[n // b]
		op2 = (6 * Y) / 5.0 + total / 5.0
		dp[n] = min(op1, op2)
		
	print("{:.15f}".format(dp[N]))

if __name__ == '__main__':
	main()