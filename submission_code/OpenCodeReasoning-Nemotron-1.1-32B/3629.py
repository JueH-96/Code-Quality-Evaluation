MOD = 10**9 + 7

class Solution:
	def lengthAfterTransformations(self, s: str, t: int) -> int:
		n_letters = 26
		
		def mat_mult(A, B, mod):
			n = len(A)
			p = len(A[0])
			m = len(B[0])
			C = [[0] * m for _ in range(n)]
			for i in range(n):
				for k in range(p):
					a_ik = A[i][k]
					for j in range(m):
						C[i][j] = (C[i][j] + a_ik * B[k][j]) % mod
			return C
		
		def matrix_power(matrix, power, mod):
			n = len(matrix)
			result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
			base = matrix
			while power:
				if power & 1:
					result = mat_mult(result, base, mod)
				base = mat_mult(base, base, mod)
				power >>= 1
			return result
		
		T = [[0] * n_letters for _ in range(n_letters)]
		for i in range(n_letters - 1):
			T[i][i + 1] = 1
		T[n_letters - 1][0] = 1
		T[n_letters - 1][1] = 1
		
		T_exp = matrix_power(T, t, MOD)
		
		f = [0] * n_letters
		for i in range(n_letters):
			total_row = 0
			for j in range(n_letters):
				total_row = (total_row + T_exp[i][j]) % MOD
			f[i] = total_row
		
		total_length = 0
		for char in s:
			idx = ord(char) - ord('a')
			total_length = (total_length + f[idx]) % MOD
		
		return total_length