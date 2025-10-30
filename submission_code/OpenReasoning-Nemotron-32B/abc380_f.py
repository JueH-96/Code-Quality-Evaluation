import sys

sys.setrecursionlimit(1000000)

def remove_one(arr, value):
	lst = list(arr)
	for i in range(len(lst)):
		if lst[i] == value:
			del lst[i]
			break
	return tuple(lst)

def add_one(arr, value):
	lst = list(arr)
	lst.append(value)
	lst.sort()
	return tuple(lst)

memo = {}

def dfs(handT, handA, table, turn):
	key = (handT, handA, table, turn)
	if key in memo:
		return memo[key]
	
	if turn == 0 and len(handT) == 0:
		memo[key] = False
		return False
	if turn == 1 and len(handA) == 0:
		memo[key] = False
		return False
		
	if turn == 0:
		current_hand = handT
		opponent_hand = handA
	else:
		current_hand = handA
		opponent_hand = handT
		
	moves = set()
	distinct_x = set(current_hand)
	
	for x in distinct_x:
		new_hand_current = remove_one(current_hand, x)
		new_table = add_one(table, x)
		if turn == 0:
			next_state = (new_hand_current, opponent_hand, new_table, 1)
		else:
			next_state = (opponent_hand, new_hand_current, new_table, 0)
		moves.add(next_state)
		
		distinct_y = set(table)
		for y in distinct_y:
			if y < x:
				new_hand_current2 = remove_one(current_hand, x)
				new_hand_current2 = add_one(new_hand_current2, y)
				new_table2 = remove_one(table, y)
				new_table2 = add_one(new_table2, x)
				if turn == 0:
					next_state2 = (new_hand_current2, opponent_hand, new_table2, 1)
				else:
					next_state2 = (opponent_hand, new_hand_current2, new_table2, 0)
				moves.add(next_state2)
				
	for state in moves:
		if not dfs(*state):
			memo[key] = True
			return True
			
	memo[key] = False
	return False

def main():
	data = sys.stdin.read().split()
	if not data:
		print("Aoki")
		return
		
	n = int(data[0])
	m = int(data[1])
	l = int(data[2])
	A = list(map(int, data[3:3+n]))
	B = list(map(int, data[3+n:3+n+m]))
	C = list(map(int, data[3+n+m:3+n+m+l]))
	
	handT = tuple(sorted(A))
	handA = tuple(sorted(B))
	table = tuple(sorted(C))
	
	if dfs(handT, handA, table, 0):
		print("Takahashi")
	else:
		print("Aoki")

if __name__ == '__main__':
	main()