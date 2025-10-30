import itertools

def solve():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, input().split())))

    def is_disappointed(permutation):
        for i in range(3):
            # Horizontal
            if permutation.index((i, 0)) < permutation.index((i, 1)) and \
               grid[i][0] == grid[i][1] and \
               permutation.index((i, 2)) > permutation.index((i, 0)) and \
               permutation.index((i, 2)) > permutation.index((i, 1)) and \
               grid[i][2] != grid[i][0]:
                return True
            
            # Vertical
            if permutation.index((0, i)) < permutation.index((1, i)) and \
               grid[0][i] == grid[1][i] and \
               permutation.index((2, i)) > permutation.index((0, i)) and \
               permutation.index((2, i)) > permutation.index((1, i)) and \
               grid[2][i] != grid[0][i]:
                return True
        
        # Diagonal 1
        if permutation.index((0, 0)) < permutation.index((1, 1)) and \
           grid[0][0] == grid[1][1] and \
           permutation.index((2, 2)) > permutation.index((0, 0)) and \
           permutation.index((2, 2)) > permutation.index((1, 1)) and \
           grid[2][2] != grid[0][0]:
            return True
        
        # Diagonal 2
        if permutation.index((2, 0)) < permutation.index((1, 1)) and \
           grid[2][0] == grid[1][1] and \
           permutation.index((0, 2)) > permutation.index((2, 0)) and \
           permutation.index((0, 2)) > permutation.index((1, 1)) and \
           grid[0][2] != grid[2][0]:
            return True
        
        return False

    cells = list(itertools.product(range(3), range(3)))
    
    count = 0
    total_permutations = 0
    
    for permutation in itertools.permutations(cells):
        total_permutations += 1
        if not is_disappointed(permutation):
            count += 1
    
    print(count / total_permutations)

solve()