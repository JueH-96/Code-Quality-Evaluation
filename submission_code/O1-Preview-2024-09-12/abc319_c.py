# YOUR CODE HERE
import sys
import itertools

import threading
def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)

    grid = [list(map(int, sys.stdin.readline().split())) for _ in range(3)]
    
    lines = []
    # Rows
    for i in range(3):
        lines.append([(i, 0), (i, 1), (i, 2)])
    # Columns
    for j in range(3):
        lines.append([(0, j), (1, j), (2, j)])
    # Diagonals
    lines.append([(0,0), (1,1), (2,2)])
    lines.append([(2,0), (1,1), (0,2)])
    
    dangerous_lines = []
    for line in lines:
        nums = [grid[i][j] for i,j in line]
        if nums[0] == nums[1] == nums[2]:
            continue  # Not possible due to constraints
        elif nums.count(nums[0]) == 2 or nums.count(nums[1]) == 2 or nums.count(nums[2]) == 2:
            dangerous_lines.append(line)
        else:
            continue  # All numbers different, not dangerous

    num_dangerous = len(dangerous_lines)
    if num_dangerous == 0:
        # No dangerous lines, probability is 1
        print(1.0)
        return

    from collections import defaultdict

    # Let's use inclusion-exclusion principle
    # Total number of permutations
    total_perms = math.factorial(9)

    def compute_prob(subset):
        # Collect squares involved in the dangerous lines in subset
        squares_set = set()
        for line in subset:
            for square in line:
                squares_set.add(square)
        squares = list(squares_set)
        square_indices = {square: idx for idx, square in enumerate(squares)}
        m = len(squares)
        total_m_perms = math.factorial(m)
        count = 0
        perms = itertools.permutations(range(m))
        for perm in perms:
            square_order = [squares[i] for i in perm]
            # For each line in subset, check if he gets disappointed
            disappointed_in_all = True
            for line in subset:
                # Get positions in perm of the squares in this line
                line_squares = line
                line_indices = [square_indices[sq] for sq in line_squares]
                line_positions = [perm.index(idx) for idx in line_indices]
                # Numbers in the squares
                nums = [grid[i][j] for i,j in line_squares]
                # Identify which squares have same numbers
                count_nums = defaultdict(int)
                for num in nums:
                    count_nums[num] +=1
                for num, cnt in count_nums.items():
                    if cnt == 2:
                        same_num = num
                        break
                else:
                    continue  # Should not happen in dangerous line
                # Squares with same number and different number
                same_squares = [square_indices[line_squares[i]] for i in range(3) if nums[i]==same_num]
                diff_square = [square_indices[line_squares[i]] for i in range(3) if nums[i]!=same_num][0]
                # Check if positions of same_squares in perm are before diff_square
                same_positions = [perm.index(idx) for idx in same_squares]
                diff_position = perm.index(diff_square)
                if max(same_positions) < diff_position:
                    continue  # He gets disappointed due to this line
                else:
                    disappointed_in_all = False
                    break  # Since he didn't get disappointed due to this line, he doesn't get disappointed in all lines of subset
            if disappointed_in_all:
                count +=1
        prob = count / total_m_perms
        # Multiply by the permutations of the remaining squares
        remaining_squares = 9 - m
        prob *= math.factorial(remaining_squares) / total_perms
        return prob

    from itertools import combinations
    P_D = 0.0
    n = len(dangerous_lines)
    for k in range(1, n+1):
        sign = (-1)**(k+1)
        for subset in combinations(dangerous_lines, k):
            prob = compute_prob(subset)
            P_D += sign * prob

    P_not_D = 1.0 - P_D
    print("{0:.30f}".format(P_not_D))
threading.Thread(target=main).start()