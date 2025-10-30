def main():
    import sys
    input_data = sys.stdin.read().split()
    if len(input_data) < 6:
        return
    # Parse input values to integers
    x_A, y_A, x_B, y_B, x_C, y_C = map(int, input_data[:6])
    
    # Function to compute squared distance between two points
    def squared_distance(x1, y1, x2, y2):
        return (x1 - x2)**2 + (y1 - y2)**2
    
    # Calculate squared side lengths
    AB2 = squared_distance(x_A, y_A, x_B, y_B)
    BC2 = squared_distance(x_B, y_B, x_C, y_C)
    CA2 = squared_distance(x_C, y_C, x_A, y_A)
    
    # Check if any combination satisfies the Pythagorean theorem
    if AB2 + BC2 == CA2 or AB2 + CA2 == BC2 or BC2 + CA2 == AB2:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    main()