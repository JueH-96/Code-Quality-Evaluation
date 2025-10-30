import sys

def main():
	data = sys.stdin.read().splitlines()
	if not data:
		print("")
		return
	S = data[0].strip()
	n = len(S)
	if n == 0:
		print("")
		return

	rev = S[::-1]
	
	base = 131
	mod1 = 10**9 + 7
	mod2 = 10**9 + 9
	
	pow1 = [1] * (n + 1)
	pow2 = [1] * (n + 1)
	for i in range(1, n + 1):
		pow1[i] = (pow1[i - 1] * base) % mod1
		pow2[i] = (pow2[i - 1] * base) % mod2

	H1_S = [0] * n
	H2_S = [0] * n
	if n > 0:
		H1_S[0] = ord(S[0]) % mod1
		H2_S[0] = ord(S[0]) % mod2
		for i in range(1, n):
			H1_S[i] = (H1_S[i - 1] * base + ord(S[i])) % mod1
			H2_S[i] = (H2_S[i - 1] * base + ord(S[i])) % mod2

	H1_rev = [0] * n
	H2_rev = [0] * n
	if n > 0:
		H1_rev[0] = ord(rev[0]) % mod1
		H2_rev[0] = ord(rev[0]) % mod2
		for i in range(1, n):
			H1_rev[i] = (H1_rev[i - 1] * base + ord(rev[i])) % mod1
			H2_rev[i] = (H2_rev[i - 1] * base + ord(rev[i])) % mod2

	def get_hash_S(l, r):
		if l == 0:
			return (H1_S[r], H2_S[r])
		h1 = (H1_S[r] - H1_S[l - 1] * pow1[r - l + 1]) % mod1
		h2 = (H2_S[r] - H2_S[l - 1] * pow2[r - l + 1]) % mod2
		if h1 < 0:
			h1 += mod1
		if h2 < 0:
			h2 += mod2
		return (h1, h2)

	def get_hash_rev(l, r):
		if l == 0:
			return (H1_rev[r], H2_rev[r])
		h1 = (H1_rev[r] - H1_rev[l - 1] * pow1[r - l + 1]) % mod1
		h2 = (H2_rev[r] - H2_rev[l - 1] * pow2[r - l + 1]) % mod2
		if h1 < 0:
			h1 += mod1
		if h2 < 0:
			h2 += mod2
		return (h1, h2)

	low, high = 1, n
	ans = 0
	while low <= high:
		mid = (low + high) // 2
		h_s = get_hash_S(n - mid, n - 1)
		h_r = get_hash_rev(0, mid - 1)
		if h_s == h_r:
			ans = mid
			low = mid + 1
		else:
			high = mid - 1

	result = S + rev[ans:]
	print(result)

if __name__ == "__main__":
	main()