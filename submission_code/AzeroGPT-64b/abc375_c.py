n = int(input())
grid = [list(map(lambda x: '#' if x=='.' else '.', list(input()))) for _ in range(n)]
size = (n//2)-1
start= 1
end = n-2

while(start <= size):
	for i in range(start,n-start):
		for j in range(end,n-start-1,-1):
			grid[i][j], grid[j][n-i-1] = grid[j][n-i-1], grid[i][j]
	start += 1
	end -= 1

[print(''.join(row)) for row in grid]