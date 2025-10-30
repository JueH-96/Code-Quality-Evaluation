def main():
	S_x, S_y = map(int, input().split())
	T_x, T_y = map(int, input().split())
	
	if S_y == T_y:
		ans = abs(S_x - T_x) // 2
	else:
		ans = abs(T_y - S_y)
		
	print(ans)

if __name__ == '__main__':
	main()