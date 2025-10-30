# YOUR CODE HERE
A, B = map(int, input().split())

def are_adjacent_horizontally(a, b):
    board = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    
    for row in board:
        if a in row and b in row:
            if abs(row.index(a) - row.index(b)) == 1:
                return True
    return False

print("Yes" if are_adjacent_horizontally(A, B) else "No")