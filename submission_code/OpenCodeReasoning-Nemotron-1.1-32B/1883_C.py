t = int(input().strip())
for _ in range(t):
	data = input().split()
	n = int(data[0])
	k = int(data[1])
	a = list(map(int, input().split()))
	
	if k == 2:
		found_even = False
		for x in a:
			if x % 2 == 0:
				found_even = True
				break
		if found_even:
			print(0)
		else:
			print(1)
			
	elif k == 3:
		min_ops = 10**9
		for x in a:
			r = x % 3
			if r == 0:
				min_ops = 0
				break
			else:
				if r == 1:
					op = 2
				else:
					op = 1
				if op < min_ops:
					min_ops = op
		print(min_ops)
		
	elif k == 4:
		total_exponent = 0
		for x in a:
			temp = x
			while temp % 2 == 0:
				total_exponent += 1
				temp //= 2
				
		if total_exponent >= 2:
			print(0)
		else:
			if total_exponent == 0:
				min_ops_div4 = 10**9
				for x in a:
					r = x % 4
					if r == 0:
						op = 0
					elif r == 1:
						op = 3
					elif r == 2:
						op = 2
					else:
						op = 1
					if op < min_ops_div4:
						min_ops_div4 = op
				ans = min(min_ops_div4, 2)
				print(ans)
			else:
				min_ops_div4 = 10**9
				for x in a:
					if x % 2 == 0:
						r = x % 4
						if r == 0:
							op = 0
						else:
							op = 2
						if op < min_ops_div4:
							min_ops_div4 = op
				ans = min(min_ops_div4, 1)
				print(ans)
				
	elif k == 5:
		min_ops = 10**9
		for x in a:
			r = x % 5
			if r == 0:
				op = 0
			else:
				op = 5 - r
			if op < min_ops:
				min_ops = op
		print(min_ops)