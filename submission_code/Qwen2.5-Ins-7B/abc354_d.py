# YOUR CODE HERE
def solve(A, B, C, D):
    def is_black(x, y):
        if (x - y) % 2 == 0:
            return (x + y) % 2 == 0
        else:
            return (x + y) % 2 != 0
    
    black_area = 0
    for x in range(A, C + 1):
        for y in range(B, D + 1):
            if is_black(x, y):
                black_area += 1
    return black_area * 2

A, B, C, D = map(int, input().split())
print(solve(A, B, C, D))