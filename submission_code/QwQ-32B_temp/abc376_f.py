import sys

def main():
    import sys
    N, Q = map(int, sys.stdin.readline().split())
    current_L = 1
    current_R = 2
    total = 0

    for _ in range(Q):
        H, T = sys.stdin.readline().split()
        T = int(T)
        if H == 'L':
            # Move L to T
            d1 = (T - current_L) % N
            d2 = (current_L - T) % N
            min_dist = min(d1, d2)
            path_is_clockwise = (d1 < d2)
            blocked = False

            if path_is_clockwise:
                # Check if current_R is on the clockwise path
                dist_R = (current_R - current_L) % N
                if dist_R <= d1:
                    blocked = True
            else:
                # Check if current_R is on the counter-clockwise path
                dist_R = (current_L - current_R) % N
                if dist_R <= d2:
                    blocked = True

            steps = min_dist + (1 if blocked else 0)
            new_L = T
            if blocked:
                if path_is_clockwise:
                    new_R = (current_R - 1) % N
                else:
                    new_R = (current_R + 1) % N
            else:
                new_R = current_R
            current_L, current_R = new_L, new_R
        else:
            # Move R to T
            d1 = (T - current_R) % N
            d2 = (current_R - T) % N
            min_dist = min(d1, d2)
            path_is_clockwise = (d1 < d2)
            blocked = False

            if path_is_clockwise:
                # Check if current_L is on the clockwise path (from current_R to T)
                dist_L = (current_L - current_R) % N
                if dist_L <= d1:
                    blocked = True
            else:
                # Check if current_L is on the counter-clockwise path
                dist_L = (current_R - current_L) % N
                if dist_L <= d2:
                    blocked = True

            steps = min_dist + (1 if blocked else 0)
            new_R = T
            if blocked:
                if path_is_clockwise:
                    new_L = (current_L - 1) % N
                else:
                    new_L = (current_L + 1) % N
            else:
                new_L = current_L
            current_L, current_R = new_L, new_R
        total += steps

    print(total)

if __name__ == "__main__":
    main()