# YOUR CODE HERE
def count_black_cells(A, B, C, D):
    def count_black_up_to(x, y):
        if x < 0:
            return count_black_up_to(-x-1, y)
        if y < 0:
            return count_black_up_to(x, -y-1)
        if x % 2 == 1 and y % 2 == 1:
            return (x // 2 + 1) * (y // 2 + 1) + (x // 2) * (y // 2)
        else:
            return (x // 2 + 1) * (y // 2 + 1) + (x // 2) * (y // 2) - 1

    total_black = count_black_up_to(C-1, D-1) - count_black_up_to(A-1, D-1) - count_black_up_to(C-1, B-1) + count_black_up_to(A-1, B-1)
    return total_black

import sys
input = sys.stdin.read
A, B, C, D = map(int, input().strip().split())
print(2 * count_black_cells(A, B, C, D))