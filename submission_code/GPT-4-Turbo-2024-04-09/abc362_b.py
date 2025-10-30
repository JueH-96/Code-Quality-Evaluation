import sys
input = sys.stdin.read

def is_right_triangle(xA, yA, xB, yB, xC, yC):
    # Calculate the squared lengths of the sides of the triangle
    AB2 = (xB - xA) ** 2 + (yB - yA) ** 2
    BC2 = (xC - xB) ** 2 + (yC - yB) ** 2
    CA2 = (xA - xC) ** 2 + (yA - yC) ** 2
    
    # Check if any permutation of these squared lengths satisfies the Pythagorean theorem
    if AB2 + BC2 == CA2 or BC2 + CA2 == AB2 or CA2 + AB2 == BC2:
        return "Yes"
    else:
        return "No"

def main():
    data = input().split()
    xA, yA = int(data[0]), int(data[1])
    xB, yB = int(data[2]), int(data[3])
    xC, yC = int(data[4]), int(data[5])
    
    result = is_right_triangle(xA, yA, xB, yB, xC, yC)
    print(result)

if __name__ == "__main__":
    main()