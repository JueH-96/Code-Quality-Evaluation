# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    A, B, C, D = map(int, data.split())
    
    def count_black(x1, y1, x2, y2):
        # Count the number of black squares in the rectangle [x1, x2) x [y1, y2)
        black_count = 0
        for x in range(x1, x2):
            for y in range(y1, y2):
                if (x + y) % 2 == 0:
                    black_count += 1
        return black_count
    
    # Calculate the number of black squares in the rectangle [A, C) x [B, D)
    black_area = count_black(A, B, C, D)
    
    # Print twice the black area
    print(2 * black_area)