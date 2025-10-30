import sys

def generate_children(state):
	turn, T, A, C = state
	children = []
	if turn == 0:
		if len(T) == 0:
			return children
		for x in set(T):
			lst_T = list(T)
			i = lst_T.index(x)
			del lst_T[i]
			new_T = tuple(lst_T)
			new_table = tuple(sorted(C + (x,)))
			children.append((1, new_T, A, new_table))
			distinct_y = set()
			for card in new_table:
				if card < x:
					distinct_y.add(card)
			for y in distinct_y:
				lst_new_table = list(new_table)
				i2 = lst_new_table.index(y)
				del lst_new_table[i2]
				new_table2 = tuple(lst_new_table)
				new_T_hand = tuple(sorted(new_T + (y,)))
				children.append((1, new_T_hand, A, new_table2))
		return children
	else:
		if len(A) == 0:
			return children
		for x in set(A):
			lst_A = list(A)
			i = lst_A.index(x)
			del lst_A[i]
			new_A = tuple(lst_A)
			new_table = tuple(sorted(C + (x,)))
			children.append((0, T, new_A, new_table))
			distinct_y = set()
			for card in new_table:
				if card < x:
					distinct_y.add(card)
			for y in distinct_y:
				lst_new_table = list(new_table)
				i2 = lst_new_table.index(y)
				del lst_new_table[i2]
				new_table2 = tuple(lst_new_table)
				new_A_hand = tuple(sorted(new_A + (y,)))
				children.append((0, T, new_A_hand, new_table2))
		return children

def main():
	data = sys.stdin.read().split()
	if not data:
		return
	it = iter(data)
	N = int(next(it))
	M = int(next(it))
	L = int(next(it))
	A = [int(next(it)) for _ in range(N)]
	B = [int(next(it)) for _ in range(M)]
	C = [int(next(it)) for _ in range(L)]
	
	T0 = tuple(sorted(A))
	A0 = tuple(sorted(B))
	C0 = tuple(sorted(C))
	
	memo = {}
	stack = [((0, T0, A0, C0), None)]
	
	while stack:
		state, children_list = stack.pop()
		if state in memo:
			continue
			
		if children_list is None:
			children_list = generate_children(state)
			all_computed = True
			for child in children_list:
				if child not in memo:
					all_computed = False
					break
					
			if all_computed:
				turn = state[0]
				if turn == 0:
					win = False
					for child in children_list:
						if not memo[child]:
							win = True
							break
					memo[state] = win
				else:
					win = False
					for child in children_list:
						if not memo[child]:
							win = True
							break
					memo[state] = win
			else:
				stack.append((state, children_list))
				for child in children_list:
					if child not in memo:
						stack.append((child, None))
		else:
			turn = state[0]
			if turn == 0:
				win = False
				for child in children_list:
					if not memo[child]:
						win = True
						break
				memo[state] = win
			else:
				win = False
				for child in children_list:
					if not memo[child]:
						win = True
						break
				memo[state] = win
				
	result = memo.get((0, T0, A0, C0), False)
	if result:
		print("Takahashi")
	else:
		print("Aoki")

if __name__ == "__main__":
	main()