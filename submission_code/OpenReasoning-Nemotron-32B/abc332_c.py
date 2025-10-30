def main():
	import sys
	data = sys.stdin.read().splitlines()
	if not data: 
		print(0)
		return
	n_m = data[0].split()
	N = int(n_m[0])
	M = int(n_m[1])
	S = data[1].strip()
	
	segments = S.split('0')
	max_req = 0
	for seg in segments:
		if seg:
			n = len(seg)
			count2 = seg.count('2')
			req = max(count2, n - M)
			if req > max_req:
				max_req = req
	print(max_req)

if __name__ == "__main__":
	main()