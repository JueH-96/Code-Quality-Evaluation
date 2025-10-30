def main():
    grid = []
    for _ in range(9):
        row = list(map(int, input().split()))
        grid.append(row)
    
    for i in range(9):
        if len(set(grid[i])) != 9:
            print("No")
            return
    
    for j in range(9):
        column = [grid[i][j] for i in range(9)]
        if len(set(column)) != 9:
            print("No")
            return
            
    starts_r = [0, 3, 6]
    starts_c = [0, 3, 6]
    for r in starts_r:
        for c in starts_c:
            nums = []
            for i in range(r, r+3):
                for j in range(c, c+3):
                    nums.append(grid[i][j])
            if len(set(nums)) != 9:
                print("No")
                return
                
    print("Yes")

if __name__ == "__main__":
    main()