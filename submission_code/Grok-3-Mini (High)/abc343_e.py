import sys

def get_ox(L, choice):
    if L > 0:
        if choice == 0:
            return 7, L, L
        elif choice == 1:
            return L, 7, L
        elif choice == 2:
            return L, L, 7
    else:  # L == 0
        if choice == 0:
            return 7, 0, 0
        elif choice == 1:
            return 0, 7, 0
        elif choice == 2:
            return 0, 0, 7
        elif choice == 3:
            return 0, 0, 0

# Read input
V1, V2, V3 = map(int, input().split())

for Lx in range(8):  # 0 to 7
    for Ly in range(8):
        for Lz in range(8):
            if Lx * Ly * Lz == V3:
                num_choice_x = 3 if Lx > 0 else 4
                num_choice_y = 3 if Ly > 0 else 4
                num_choice_z = 3 if Lz > 0 else 4
                for choice_x in range(num_choice_x):
                    for choice_y in range(num_choice_y):
                        for choice_z in range(num_choice_z):
                            ox = get_ox(Lx, choice_x)
                            oy = get_ox(Ly, choice_y)
                            oz = get_ox(Lz, choice_z)
                            V12 = ox[0] * oy[0] * oz[0]
                            V13 = ox[1] * oy[1] * oz[1]
                            V23 = ox[2] * oy[2] * oz[2]
                            S2_calc = V12 + V13 + V23
                            if S2_calc == V2 + 3 * V3:
                                # Set positions for x
                                if choice_x < 3:
                                    if choice_x == 0:
                                        ax = [0, 0, Lx - 7]
                                    elif choice_x == 1:
                                        ax = [0, Lx - 7, 0]
                                    elif choice_x == 2:
                                        ax = [Lx - 7, 0, 0]
                                else:  # choice_x == 3
                                    ax = [0, 7, 14]
                                # Set positions for y
                                if choice_y < 3:
                                    if choice_y == 0:
                                        ay = [0, 0, Ly - 7]
                                    elif choice_y == 1:
                                        ay = [0, Ly - 7, 0]
                                    elif choice_y == 2:
                                        ay = [Ly - 7, 0, 0]
                                else:
                                    ay = [0, 7, 14]
                                # Set positions for z
                                if choice_z < 3:
                                    if choice_z == 0:
                                        az = [0, 0, Lz - 7]
                                    elif choice_z == 1:
                                        az = [0, Lz - 7, 0]
                                    elif choice_z == 2:
                                        az = [Lz - 7, 0, 0]
                                else:
                                    az = [0, 7, 14]
                                # Extract coordinates for each cube
                                x1, x2, x3 = ax[0], ax[1], ax[2]
                                y1, y2, y3 = ay[0], ay[1], ay[2]
                                z1, z2, z3 = az[0], az[1], az[2]
                                # Output the result
                                print("Yes")
                                print(f"{x1} {y1} {z1} {x2} {y2} {z2} {x3} {y3} {z3}")
                                sys.exit()
# If no configuration found
print("No")