import math
from itertools import permutations

class Solution:
	def countGoodIntegers(self, n: int, k: int) -> int:
		fact = [1] * 11
		for i in range(1, 11):
			fact[i] = fact[i-1] * i
		
		ans = 0
		for a0 in range(0, n+1):
			for a1 in range(0, n+1 - a0):
				for a2 in range(0, n+1 - a0 - a1):
					for a3 in range(0, n+1 - a0 - a1 - a2):
						for a4 in range(0, n+1 - a0 - a1 - a2 - a3):
							for a5 in range(0, n+1 - a0 - a1 - a2 - a3 - a4):
								for a6 in range(0, n+1 - a0 - a1 - a2 - a3 - a4 - a5):
									for a7 in range(0, n+1 - a0 - a1 - a2 - a3 - a4 - a5 - a6):
										for a8 in range(0, n+1 - a0 - a1 - a2 - a3 - a4 - a5 - a6 - a7):
											a9 = n - a0 - a1 - a2 - a3 - a4 - a5 - a6 - a7 - a8
											F = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9]
											
											non_zero = False
											for d in range(1, 10):
												if F[d] > 0:
													non_zero = True
													break
											if not non_zero:
												continue
											
											odd_count = 0
											center_digit = -1
											for d in range(10):
												if F[d] % 2 == 1:
													odd_count += 1
													center_digit = d
											if n % 2 == 0:
												if odd_count != 0:
													continue
											else:
												if odd_count != 1:
													continue
							
											m = (n + 1) // 2
											G = [0] * 10
											if n % 2 == 0:
												for d in range(10):
													G[d] = F[d] // 2
											else:
												for d in range(10):
													if d == center_digit:
														G[d] = (F[d] + 1) // 2
													else:
														G[d] = F[d] // 2
											G_list = []
											for d in range(10):
												G_list.extend([d] * G[d])
											if len(G_list) != m:
												continue
							
											found = False
											for p in permutations(G_list):
												if p[0] == 0:
													continue
												if n % 2 == 1:
													if p[-1] != center_digit:
														continue
												if n % 2 == 0:
													s1 = ''.join(str(x) for x in p)
													s2 = s1[::-1]
													num_str = s1 + s2
												else:
													s1 = ''.join(str(x) for x in p[:-1])
													center_char = str(p[-1])
													s2 = s1[::-1]
													num_str = s1 + center_char + s2
												num = int(num_str)
												if num % k == 0:
													found = True
													break
							
											if not found:
												continue
							
											denom = 1
											for count in F:
												denom *= fact[count]
											total_permutations = fact[n] // denom
							
											if F[0] > 0:
												denom0 = fact[F[0] - 1]
												for d in range(1, 10):
													denom0 *= fact[F[d]]
												count0 = fact[n-1] // denom0
											else:
												count0 = 0
							
											count_good = total_permutations - count0
											ans += count_good
		return ans