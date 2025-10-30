def main():
	n = int(input().strip())
	a_list = list(map(int, input().split()))
	
	state0 = 0
	state1 = -10**18
	
	for a in a_list:
		new_state0 = max(state0, state1 + 2 * a)
		new_state1 = max(state1, state0 + a)
		state0, state1 = new_state0, new_state1
		
	print(max(state0, state1))

if __name__ == "__main__":
	main()