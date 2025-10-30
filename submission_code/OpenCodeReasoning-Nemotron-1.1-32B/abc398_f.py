MOD1 = 10**9 + 7
MOD2 = 10**9 + 9
BASE1 = 131
BASE2 = 13131

import sys

def main():
	S = sys.stdin.readline().strip()
	n = len(S)
	if n == 0:
		print("")
		return
	
	rev = S[::-1]
	
	pow1_1 = [1] * n
	pow1_2 = [1] * n
	for i in range(1, n):
		pow1_1[i] = (pow1_1[i-1] * BASE1) % MOD1
		pow1_2[i] = (pow1_2[i-1] * BASE2) % MOD2

	H_left1 = [0] * (n+1)
	H_left2 = [0] * (n+1)
	for i in range(1, n+1):
		char_val = ord(S[i-1]) - 64
		H_left1[i] = (H_left1[i-1] + char_val * pow1_1[i-1]) % MOD1
		H_left2[i] = (H_left2[i-1] + char_val * pow1_2[i-1]) % MOD2

	inv_base1 = pow(BASE1, MOD1-2, MOD1)
	inv_base2 = pow(BASE2, MOD2-2, MOD2)
	
	inv_pow1 = [1] * (n+1)
	inv_pow2 = [1] * (n+1)
	for i in range(1, n+1):
		inv_pow1[i] = (inv_pow1[i-1] * inv_base1) % MOD1
		inv_pow2[i] = (inv_pow2[i-1] * inv_base2) % MOD2

	H_rev1 = [0] * (n+1)
	H_rev2 = [0] * (n+1)
	for i in range(1, n+1):
		char_val = ord(rev[i-1]) - 64
		H_rev1[i] = (H_rev1[i-1] + char_val * pow1_1[i-1]) % MOD1
		H_rev2[i] = (H_rev2[i-1] + char_val * pow1_2[i-1]) % MOD2

	i = 0
	for i_candidate in range(0, n+1):
		len_suffix = n - i_candidate
		hash1_S = (H_left1[n] - H_left1[i_candidate]) % MOD1
		if hash1_S < 0:
			hash1_S += MOD1
		hash1_S = (hash1_S * inv_pow1[i_candidate]) % MOD1
		
		hash2_S = (H_left2[n] - H_left2[i_candidate]) % MOD2
		if hash2_S < 0:
			hash2_S += MOD2
		hash2_S = (hash2_S * inv_pow2[i_candidate]) % MOD2
		
		hash1_rev = H_rev1[len_suffix]
		hash2_rev = H_rev2[len_suffix]
		
		if hash1_S == hash1_rev and hash2_S == hash2_rev:
			i = i_candidate
			break

	prefix_part = S[:i]
	result = S + prefix_part[::-1]
	print(result)

if __name__ == '__main__':
	main()