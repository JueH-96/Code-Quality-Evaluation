import sys

def calculate_area(sheets):
    # Initialize a set to store the covered points
    covered_points = set()

    # Iterate over each sheet
    for sheet in sheets:
        # Extract the coordinates of the sheet
        ax, bx, ay, by = sheet

        # Iterate over the x and y coordinates of the sheet
        for x in range(ax, bx):
            for y in range(ay, by):
                # Add the point to the set of covered points
                covered_points.add((x, y))

    # Return the area of the covered region
    return len(covered_points)

def main():
    # Read the number of sheets
    n = int(sys.stdin.readline().strip())

    # Read the coordinates of each sheet
    sheets = []
    for _ in range(n):
        ax, bx, ay, by = map(int, sys.stdin.readline().strip().split())
        sheets.append((ax, bx, ay, by))

    # Calculate and print the area of the covered region
    area = calculate_area(sheets)
    print(area)

if __name__ == "__main__":
    main()