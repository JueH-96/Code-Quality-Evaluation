def main():
    import sys
    from itertools import permutations
    # Read the input grid as a list of integers.
    data = sys.stdin.read().split()
    if not data:
        return
    grid = [int(x) for x in data]  # grid is of length 9; index mapping: row i, col j -> index = 3*i+j
    
    # Define the 8 lines (each line is a tuple of 3 indices)
    lines = []
    # Horizontal lines (rows)
    for i in range(3):
        lines.append((3*i, 3*i+1, 3*i+2))
    # Vertical lines (columns)
    for j in range(3):
        lines.append((j, j+3, j+6))
    # Diagonals
    lines.append((0, 4, 8))
    lines.append((2, 4, 6))
    
    # In a random order (permutation) Takahashi sees all 9 cells.
    # He gets disappointed if there exists a line (one of the 8) for which,
    # the two cells he sees first in that line (according to his overall revealing order)
    # have the same number and the third (last seen among those three)
    # has a different number.
    #
    # Our plan: iterate over all 9! orders (362880 orders) of the cells, and 
    # check for each order if any line yields the "disappointing" condition. 
    #
    # For each permutation perm (a tuple listing cell indices in order of revelation),
    # we precompute an array pos such that pos[i] is the index in perm (i.e. reveal order) 
    # of cell i. Then for each line (say indices a, b, c), we form a list:
    #   [(pos[a], grid[a]), (pos[b], grid[b]), (pos[c], grid[c])]
    # and sort it by the reveal order. If the first two numbers are equal and the third is different,
    # this permutation causes disappointment.
    
    total = 0
    success = 0
    # Iterate over all permutations of the 9 cells.
    for perm in permutations(range(9)):
        total += 1
        pos = [0] * 9
        for order, cell in enumerate(perm):
            pos[cell] = order
        
        disappointed = False
        for line in lines:
            a, b, c = line
            # Create a list of (order, number) for the three cells in this line.
            trio = [(pos[a], grid[a]), (pos[b], grid[b]), (pos[c], grid[c])]
            trio.sort(key=lambda x: x[0])
            # If the first two seen cells have the same number and the third is different, it's a disappointment.
            if trio[0][1] == trio[1][1] and trio[2][1] != trio[0][1]:
                disappointed = True
                break
        if not disappointed:
            success += 1

    probability = success / total
    # Print the probability with sufficient precision.
    sys.stdout.write("{:.12f}".format(probability))

if __name__ == '__main__':
    main()