# Define the cell class to store the cell information
Cell = collections.namedtuple('Cell', ('row', 'col'))

# The function to check if the combination of the two matrices results in the target matrix
# Args:
# - startA: A Cell object representing the top-left corner of matrix A's overlay on matrix C
# - startB: A Cell object representing the top-left corner of matrix B's overlay on matrix C
# - matrixA, matrixB, matrixX: Matrices represented as a list of strings
# - visitedA, visitedB: Visited matrices, same as the input matrices but converted to sets for easy manipulation
# - k: An integer representing the size of the matrix
# Returns True if the overlay of matrix A and B on matrix C creates matrix X, otherwise returns False
def checkOverlay(startA, startB, matrixA, matrixB, matrixX, visitedA, visitedB, k):
    matrixACells = set()
    for cell in visitedA:
        matrixACells.add((cell.row + startA.row, cell.col + startA.col))
    for cell in visitedB:
        matrixACells.add((cell.row + startB.row, cell.col + startB.col))
    
    # Verify if the black cells in matrix A and B, when put on matrix C, match matrix X
    for r in range(k):
        for c in range(k):
            indexA = (r, c) in matrixACells
            indexX = matrixX[r][c] == '#'
            if indexA != indexX:
                return False
    return True

# The main function to solve the problem
def canFormTargetMatrix(matrixA, matrixB, matrixX):
    H_A, W_A = len(matrixA), len(matrixA[0])
    H_B, W_B = len(matrixB), len(matrixB[0])
    k = len(matrixX)
    
    visitedA = set()
    visitedB = set()
    
    # Identify the black cells in matrix A and B
    for r in range(H_A):
        for c in range(W_A):
            if matrixA[r][c] == '#':
                visitedA.add(Cell(r, c))
    for r in range(H_B):
        for c in range(W_B):
            if matrixB[r][c] == '#':
                visitedB.add(Cell(r, c))
    
    # Calculate all possible overlay combinations of matrix A and B with matrix X
    for shiftA in range(H_A - (k - 1)):
        for shiftB in range(W_A - (k - 1)):
            for shiftC in range(H_B - (k - 1)):
                for shiftD in range(W_B - (k - 1)):
                    # Check if the current overlay position results in the target matrix X
                    if checkOverlay(Cell(shiftA, shiftB), Cell(shiftC, shiftD), matrixA, matrixB, matrixX, visitedA, visitedB, k):
                        return True
    return False


# Parsing the input and invoking the main function
if __name__ == '__main__':
    H_A, W_A = map(int, input().split())
    matrixA = [input() for _ in range(H_A)]
    H_B, W_B = map(int, input().split())
    matrixB = [input() for _ in range(H_B)]
    k, _ = map(int, input().split())
    matrixX = [input() for _ in range(k)]

    # Checking if matrix X can be created by overlaying matrix A and B
    if canFormTargetMatrix(matrixA, matrixB, matrixX):
        print('Yes')
    else:
        print('No')