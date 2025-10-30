import sys

def solve():
    # Read the coordinates of the three points
    points = []
    for _ in range(3):
        x, y = map(int, sys.stdin.readline().split())
        points.append((x, y))

    A = points[0]
    B = points[1]
    C = points[2]

    # Helper function to calculate the dot product of two 2D vectors
    # A vector from P1 to P2 is (P2[0]-P1[0], P2[1]-P1[1])
    # The dot product of vector v1=(x1, y1) and v2=(x2, y2) is x1*x2 + y1*y2
    def calculate_dot_product(v1_x, v1_y, v2_x, v2_y):
        return v1_x * v2_x + v1_y * v2_y

    # Check for a right angle at vertex A
    # Vectors are AB and AC
    # Vector AB components: (B[0]-A[0], B[1]-A[1])
    # Vector AC components: (C[0]-A[0], C[1]-A[1])
    ab_x, ab_y = B[0] - A[0], B[1] - A[1]
    ac_x, ac_y = C[0] - A[0], C[1] - A[1]
    if calculate_dot_product(ab_x, ab_y, ac_x, ac_y) == 0:
        print("Yes")
        return

    # Check for a right angle at vertex B
    # Vectors are BA and BC
    # Vector BA components: (A[0]-B[0], A[1]-B[1])
    # Vector BC components: (C[0]-B[0], C[1]-B[1])
    ba_x, ba_y = A[0] - B[0], A[1] - B[1]
    bc_x, bc_y = C[0] - B[0], C[1] - B[1]
    if calculate_dot_product(ba_x, ba_y, bc_x, bc_y) == 0:
        print("Yes")
        return

    # Check for a right angle at vertex C
    # Vectors are CA and CB
    # Vector CA components: (A[0]-C[0], A[1]-C[1])
    # Vector CB components: (B[0]-C[0], B[1]-C[1])
    ca_x, ca_y = A[0] - C[0], A[1] - C[1]
    cb_x, cb_y = B[0] - C[0], B[1] - C[1]
    if calculate_dot_product(ca_x, ca_y, cb_x, cb_y) == 0:
        print("Yes")
        return

    # If none of the angles are 90 degrees, it's not a right triangle
    print("No")

# Call the solve function to execute the program
solve()