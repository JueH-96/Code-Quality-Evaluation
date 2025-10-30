mod = 998244353

def main():
	import sys
	s = sys.stdin.readline().strip()
	n = len(s)
	
	dp0 = 1
	dp1 = [0] * 26
	dp2 = [0] * 26
	dp3 = 0
	
	for i in range(n):
		char = s[i]
		if char != '?':
			new_dp0 = 0
			new_dp1 = [0] * 26
			new_dp2 = [0] * 26
			new_dp3 = 0
			
			if 'A' <= char <= 'Z':
				c_index = ord(char) - ord('A')
				new_dp0 = (new_dp0 + dp0) % mod
				new_dp1[c_index] = (new_dp1[c_index] + dp0) % mod
			else:
				new_dp0 = (new_dp0 + dp0) % mod
			
			for letter in range(26):
				count = dp1[letter]
				if count == 0:
					continue
				if 'A' <= char <= 'Z':
					c_index = ord(char) - ord('A')
					new_dp1[letter] = (new_dp1[letter] + count) % mod
					if c_index == letter:
						new_dp2[letter] = (new_dp2[letter] + count) % mod
					else:
						new_dp1[c_index] = (new_dp1[c_index] + count) % mod
				else:
					new_dp1[letter] = (new_dp1[letter] + count) % mod
			
			for letter in range(26):
				count = dp2[letter]
				if count == 0:
					continue
				if 'a' <= char <= 'z':
					new_dp2[letter] = (new_dp2[letter] + count) % mod
					new_dp3 = (new_dp3 + count) % mod
				else:
					c_index = ord(char) - ord('A')
					new_dp2[letter] = (new_dp2[letter] + count) % mod
					new_dp1[c_index] = (new_dp1[c_index] + count) % mod
			
			new_dp3 = (new_dp3 + dp3) % mod
			
			dp0, dp1, dp2, dp3 = new_dp0, new_dp1, new_dp2, new_dp3
		else:
			new_dp0 = 0
			new_dp1 = [0] * 26
			new_dp2 = [0] * 26
			new_dp3 = 0
			
			new_dp0 = (new_dp0 + dp0 * 52) % mod
			for c_index in range(26):
				new_dp1[c_index] = (new_dp1[c_index] + dp0) % mod
			
			total_state1 = sum(dp1) % mod
			for letter in range(26):
				count = dp1[letter]
				if count != 0:
					new_dp1[letter] = (new_dp1[letter] + count * 52) % mod
					new_dp2[letter] = (new_dp2[letter] + count) % mod
			for c_index in range(26):
				add = (total_state1 - dp1[c_index]) % mod
				new_dp1[c_index] = (new_dp1[c_index] + add) % mod
			
			total_state2 = sum(dp2) % mod
			for letter in range(26):
				count = dp2[letter]
				if count != 0:
					new_dp2[letter] = (new_dp2[letter] + count * 52) % mod
					new_dp3 = (new_dp3 + count * 26) % mod
			for c_index in range(26):
				new_dp1[c_index] = (new_dp1[c_index] + total_state2) % mod
			
			new_dp3 = (new_dp3 + dp3 * 52) % mod
			
			dp0, dp1, dp2, dp3 = new_dp0, new_dp1, new_dp2, new_dp3
	
	total_dp1 = sum(dp1) % mod
	total_dp2 = sum(dp2) % mod
	ans = (dp0 + total_dp1 + total_dp2 + dp3) % mod
	print(ans)

if __name__ == "__main__":
	main()