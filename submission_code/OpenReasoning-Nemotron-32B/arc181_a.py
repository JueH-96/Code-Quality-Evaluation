import sys
import bisect

def main():
	data = sys.stdin.read().split()
	t = int(data[0])
	index = 1
	out = []
	for _ in range(t):
		n = int(data[index])
		index += 1
		p = list(map(int, data[index:index + n]))
		index += n
		
		b = [-x for x in p]
		dp = []
		for x in b:
			if not dp or x > dp[-1]:
				dp.append(x)
			else:
				pos = bisect.bisect_left(dp, x)
				dp[pos] = x
		lds_length = len(dp)
		ans = lds_length - 1
		out.append(str(ans))
	
	print("
".join(out))

if __name__ == "__main__":
	main()