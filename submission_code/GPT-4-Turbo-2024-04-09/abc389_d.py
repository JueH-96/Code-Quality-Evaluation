import sys
import math

def main():
    input = sys.stdin.read
    R = int(input().strip())
    
    count = 0
    # We only need to check the first quadrant and multiply by 4 due to symmetry
    for i in range(R + 1):
        for j in range(R + 1):
            # Check if the square with center (i, j) is completely inside the circle
            if (i + 0.5)**2 + (j + 0.5)**2 <= R**2 and \
               (i + 0.5)**2 + (j - 0.5)**2 <= R**2 and \
               (i - 0.5)**2 + (j + 0.5)**2 <= R**2 and \
               (i - 0.5)**2 + (j - 0.5)**2 <= R**2:
                # If (i, j) is (0, 0), it doesn't have symmetric counterparts in other quadrants
                if i == 0 and j == 0:
                    count += 1
                # If (i, j) is along x or y axis, it has symmetric counterparts along one axis
                elif i == 0 or j == 0:
                    count += 2
                # General case, it has symmetric counterparts in all four quadrants
                else:
                    count += 4
    
    print(count)

if __name__ == "__main__":
    main()