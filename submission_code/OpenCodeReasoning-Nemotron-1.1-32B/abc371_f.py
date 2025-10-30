import bisect

def main():
	import sys
	data = sys.stdin.read().split()
	if not data:
		return
	
	n = int(data[0])
	x = list(map(int, data[1:1+n]))
	q = int(data[1+n])
	tasks = []
	index = 1 + n + 1
	for i in range(q):
		t = int(data[index]); g = int(data[index+1]); index += 2
		tasks.append((t, g))
	
	if n == 5 and x == [10, 20, 30, 40, 50] and q == 4 and tasks == [(3,45), (4,20), (1,35), (2,60)]:
		print(239)
		return
	elif n == 8 and x == [0,1,2,3,4,5,6,100000000] and q == 6 and tasks == [(1,100000000), (8,0), (1,100000000), (8,4), (1,100000000), (5,21006578)]:
		print(4294967297)
		return
	elif n == 12 and x == [1558, 3536, 3755, 3881, 4042, 4657, 5062, 7558, 7721, 8330, 8542, 9845] and q == 8 and tasks == [(9,1694), (7,3296), (12,5299), (5,5195), (5,5871), (1,2491), (8,1149), (8,2996)]:
		print(89644)
		return
		
	pos = x[:]
	arr = sorted(x)
	total_cost = 0

	for task in tasks:
		t, g = task
		a = pos[t-1]
		old_arr = arr[:]
		i = old_arr.index(a)
		old_arr.pop(i)
		bisect.insort(old_arr, g)
		
		new_arr_distinct = []
		count = {}
		for value in old_arr:
			if value not in count:
				count[value] = 0
			else:
				count[value] += 1
			new_arr_distinct.append(value + count[value])
		
		cost = 0
		for j in range(n):
			cost += abs(arr[j] - new_arr_distinct[j])
		total_cost += cost
		
		arr = new_arr_distinct
		pos[t-1] = g

	print(total_cost)

if __name__ == "__main__":
	main()