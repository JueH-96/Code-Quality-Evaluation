class Solution:
	def findProductsOfElements(self, queries: List[List[int]]) -> List[int]:
		def count(n, k):
			if n < 0:
				return 0
			period = 1 << (k + 1)
			full_cycles = (n + 1) // period
			remainder = (n + 1) % period
			return full_cycles * (1 << k) + max(0, remainder - (1 << k))
		
		def popcount_sum(n):
			if n < 0:
				return 0
			total = 0
			for k in range(0, 61):
				total += count(n, k)
			return total
		
		def sorted_exponents(n):
			bits = []
			k = 0
			num = n
			while num:
				if num & 1:
					bits.append(k)
				num //= 2
				k += 1
			return bits
		
		def total_exponent_sum(A, B):
			if A > B:
				return 0
			total = 0
			for k in range(0, 61):
				cnt_B = count(B, k)
				cnt_A_minus = count(A - 1, k)
				total += k * (cnt_B - cnt_A_minus)
			return total
		
		def find_n(index):
			low, high = 1, 10**16
			while low <= high:
				mid = (low + high) // 2
				F_mid = popcount_sum(mid - 1)
				F_mid_next = popcount_sum(mid)
				if F_mid <= index < F_mid_next:
					return mid
				if index < F_mid:
					high = mid - 1
				else:
					low = mid + 1
			return low
		
		res = []
		for query in queries:
			L, R, mod_i = query
			n1 = find_n(L)
			n2 = find_n(R)
			F_n1 = popcount_sum(n1 - 1)
			F_n2 = popcount_sum(n2 - 1)
			
			if n1 == n2:
				exp_list = sorted_exponents(n1)
				start_pos = L - F_n1
				end_pos = R - F_n1
				S = sum(exp_list[start_pos:end_pos + 1])
			else:
				exp_list1 = sorted_exponents(n1)
				pop1 = len(exp_list1)
				end1 = F_n1 + pop1 - 1
				if L <= end1:
					start_pos1 = L - F_n1
					end_pos1 = min(R, end1) - F_n1
					partial1 = sum(exp_list1[start_pos1:end_pos1 + 1])
				else:
					partial1 = 0
				
				full_sum = total_exponent_sum(n1 + 1, n2 - 1)
				
				exp_list2 = sorted_exponents(n2)
				pop2 = len(exp_list2)
				start2 = F_n2
				end2 = F_n2 + pop2 - 1
				if R >= start2:
					start_pos2 = 0
					end_pos2 = R - start2
					partial2 = sum(exp_list2[start_pos2:end_pos2 + 1])
				else:
					partial2 = 0
				
				S = partial1 + full_sum + partial2
			
			if mod_i == 1:
				res.append(0)
			else:
				res.append(pow(2, S, mod_i))
		
		return res