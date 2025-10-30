# YOUR CODE HERE
def are_adjacent_horizontally(A, B):
    # The board is represented as:
    # 1 2 3
    # 4 5 6
    # 7 8 9
    # We need to check if A and B are in the same row and adjacent
    
    # Calculate the row index for A and B
    row_A = (A - 1) // 3
    row_B = (B - 1) // 3
    
    # Check if they are in the same row
    if row_A == row_B:
        # Check if they are adjacent
        if abs(A - B) == 1:
            return "Yes"
    
    return "No"

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip()
    A, B = map(int, data.split())
    result = are_adjacent_horizontally(A, B)
    print(result)