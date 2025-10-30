from fractions import Fraction

n = int(input())
x = [Fraction(int(val)) for val in input().split()]

while True:
    improved = False
    for i in range(n - 3):
        if x[i] + x[i + 3] < x[i + 1] + x[i + 2]:
            # Apply operation
            m = (x[i] + x[i + 3]) / 2
            new_pos1 = 2 * m - x[i + 1]
            new_pos2 = 2 * m - x[i + 2]
            x[i + 1] = new_pos1
            x[i + 2] = new_pos2
            # Ensure order is maintained
            if x[i + 1] > x[i + 2]:
                x[i + 1], x[i + 2] = x[i + 2], x[i + 1]
            improved = True
            break
    
    if not improved:
        break

print(int(sum(x)))