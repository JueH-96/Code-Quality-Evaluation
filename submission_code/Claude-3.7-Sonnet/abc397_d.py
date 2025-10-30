import math

def find_pair(N):
    for d in range(1, int(math.sqrt(N)) + 1):
        if N % d == 0:
            # Check if the smaller divisor gives a solution
            m = N // d
            disc = -3 * d * d + 12 * m
            if disc >= 0:
                Z = int(math.sqrt(disc))
                if Z * Z == disc and (Z - 3 * d) % 6 == 0 and Z > 3 * d:
                    y = (-3 * d + Z) // 6
                    x = y + d
                    if x > 0 and y > 0:
                        return (x, y)
            
            # Check if the larger divisor gives a solution
            d_complement = N // d
            if d != d_complement:  # Avoid duplicates
                disc_complement = -3 * d_complement * d_complement + 12 * d
                if disc_complement >= 0:
                    Z_complement = int(math.sqrt(disc_complement))
                    if Z_complement * Z_complement == disc_complement and (Z_complement - 3 * d_complement) % 6 == 0 and Z_complement > 3 * d_complement:
                        y_complement = (-3 * d_complement + Z_complement) // 6
                        x_complement = y_complement + d_complement
                        if x_complement > 0 and y_complement > 0:
                            return (x_complement, y_complement)
    
    return (-1, -1)

N = int(input())
result = find_pair(N)
if result[0] == -1:
    print(-1)
else:
    print(result[0], result[1])