mod = 10**9 + 7

def compute_A(a, k, d, mod):
	n = len(a)
	if k == 0:
		return 1
	dp0 = [0] * (k + 1)
	dp1 = [0] * (k + 1)
	dp0[0] = 1
	for i in range(n):
		new_dp0 = [0] * (k + 1)
		new_dp1 = [0] * (k + 1)
		for j in range(k + 1):
			new_dp0[j] = (new_dp0[j] + dp0[j] + dp1[j]) % mod
			if j >= 1:
				if i == 0:
					new_dp1[j] = (new_dp1[j] + dp0[j - 1]) % mod
				else:
					gap_val = a[i] - a[i - 1]
					if gap_val < d:
						new_dp1[j] = (new_dp1[j] + dp0[j - 1]) % mod
					else:
						new_dp1[j] = (new_dp1[j] + dp0[j - 1] + dp1[j - 1]) % mod
		dp0, dp1 = new_dp0, new_dp1
	return (dp0[k] + dp1[k]) % mod

class Solution:
	def sumOfPowers(self, nums: List[int], k: int) -> int:
		n = len(nums)
		if n < k:
			return 0
		a = sorted(nums)
		if n == 1:
			return 0
		gaps = [a[i + 1] - a[i] for i in range(n - 1)]
		max_gap = max(gaps) if gaps else 0
		critical_d = set()
		critical_d.add(0)
		critical_d.add(max_gap + 1)
		critical_d.add(max_gap + 2)
		for g in gaps:
			critical_d.add(g)
			critical_d.add(g + 1)
		critical_d = sorted(critical_d)
		A_dict = {}
		for d_val in critical_d:
			if d_val > max_gap + 1:
				continue
			A_dict[d_val] = compute_A(a, k, d_val, mod)
		total_ans = 0
		for i in range(len(critical_d) - 1):
			d_low = critical_d[i]
			d_high = critical_d[i + 1]
			if d_low > max_gap + 1:
				break
			low_bound = max(1, d_low)
			high_bound = min(max_gap + 1, d_high - 1)
			if low_bound > high_bound:
				continue
			count = high_bound - low_bound + 1
			total_ans = (total_ans + count * A_dict[d_low]) % mod
		return total_ans