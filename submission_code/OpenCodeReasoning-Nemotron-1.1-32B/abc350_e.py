import collections
from decimal import Decimal

def main():
	data = input().split()
	N = int(data[0])
	A = int(data[1])
	X = Decimal(data[2])
	Y = Decimal(data[3])
	
	if N == 0:
		print("{:.15f}".format(0.0))
		return
		
	divisors = set([A, 2, 3, 4, 5, 6])
	
	products = set()
	queue = collections.deque()
	products.add(1)
	queue.append(1)
	
	while queue:
		p = queue.popleft()
		for d in divisors:
			new_p = p * d
			if new_p <= N and new_p not in products:
				products.add(new_p)
				queue.append(new_p)
				
	states = set()
	states.add(0)
	for p in products:
		q = N // p
		states.add(q)
		
	states_sorted = sorted(states)
	
	f = {}
	for q in states_sorted:
		if q == 0:
			f[q] = Decimal(0)
		else:
			next1 = q // A
			cost1 = X + f[next1]
			
			total = Decimal(0)
			for d in [2, 3, 4, 5, 6]:
				next_d = q // d
				total += f[next_d]
				
			cost2 = (Decimal(6) * Y + total) / Decimal(5)
			
			f[q] = min(cost1, cost2)
			
	result = f[N]
	print("{:.15f}".format(result))

if __name__ == '__main__':
	main()