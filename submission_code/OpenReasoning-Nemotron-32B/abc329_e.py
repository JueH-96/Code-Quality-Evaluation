import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		return
	n, m = map(int, data[0].split())
	s = data[1].strip()
	t = data[2].strip()
	
	if n == 0:
		print("Yes")
		return
		
	dp = [set() for _ in range(n+1)]
	dp[0].add('')
	
	for i in range(n):
		for state in dp[i]:
			if i >= m - 1:
				start_index = i - m + 1
				if s[start_index:i+1] == t:
					new_state = t[1:] if m > 1 else ''
					if len(new_state) > m - 1:
						new_state = new_state[-(m-1):]
					dp[i+1].add(new_state)
			new_state2 = state + '#'
			if len(new_state2) > m - 1:
				new_state2 = new_state2[-(m-1):]
			dp[i+1].add(new_state2)
	
	if n >= m - 1:
		expected_state = s[n - m + 1:n]
	else:
		expected_state = s
		
	found = expected_state in dp[n]
	if found:
		for c in expected_state:
			if c == '#':
				found = False
				break
				
	print("Yes" if found else "No")

if __name__ == '__main__':
	main()