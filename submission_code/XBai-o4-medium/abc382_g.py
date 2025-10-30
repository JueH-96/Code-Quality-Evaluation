import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    T = int(input[ptr])
    ptr += 1
    for _ in range(T):
        K = int(input[ptr])
        Sx = int(input[ptr+1])
        Sy = int(input[ptr+2])
        Tx = int(input[ptr+3])
        Ty = int(input[ptr+4])
        ptr += 5
        
        # Compute i_s
        numerator_sx = 2 * Sx + 1
        denominator = 2 * K
        i_s = numerator_sx // denominator
        
        # Compute j_s
        numerator_sy = 2 * Sy + 1
        j_s = numerator_sy // denominator
        
        # Compute i_t
        numerator_tx = 2 * Tx + 1
        i_t = numerator_tx // denominator
        
        # Compute j_t
        numerator_ty = 2 * Ty + 1
        j_t = numerator_ty // denominator
        
        delta_i = i_t - i_s
        delta_j = j_t - j_s
        
        if delta_i == 0 and delta_j == 0:
            print(0)
            continue
        
        dx = abs(delta_i)
        dy = abs(delta_j)
        
        if delta_i == 0:
            # handle y only
            parity_s = (i_s + j_s) % 2
            if parity_s == 0:
                # vertical, need steps for y
                direction_y = 1 if delta_j > 0 else -1
                k_y = Sy - j_s * K
                if direction_y == 1:
                    steps_start = (K-1) - k_y
                else:
                    steps_start = k_y
            else:
                # horizontal, no steps
                steps_start = 0
            total = steps_start + dx + dy
        elif delta_j == 0:
            # handle x only
            parity_s = (i_s + j_s) % 2
            if parity_s == 1:
                # horizontal, need steps for x
                direction_x = 1 if delta_i > 0 else -1
                k_x = Sx - i_s * K
                if direction_x == 1:
                    steps_start = (K-1) - k_x
                else:
                    steps_start = k_x
            else:
                # vertical, no steps
                steps_start = 0
            total = steps_start + dx + dy
        else:
            # handle both
            parity_s = (i_s + j_s) % 2
            # compute steps_x_start
            if parity_s == 0:
                steps_x_start = 0
            else:
                direction_x = 1 if delta_i > 0 else -1
                k_x_start = Sx - i_s * K
                if direction_x == 1:
                    steps_x_start = (K-1) - k_x_start
                else:
                    steps_x_start = k_x_start
            # compute steps_y_start
            if parity_s == 0:
                direction_y = 1 if delta_j > 0 else -1
                k_y_start = Sy - j_s * K
                if direction_y == 1:
                    steps_y_start = (K-1) - k_y_start
                else:
                    steps_y_start = k_y_start
            else:
                steps_y_start = 0
            # compute option1 and option2
            option1 = steps_x_start + 0
            if delta_i > 0:
                steps_x_new = (K-1) - k_x_start
            else:
                steps_x_new = k_x_start
            option2 = steps_y_start + steps_x_new
            steps_start = min(option1, option2)
            total = steps_start + dx + dy
        print(total)
        
if __name__ == '__main__':
    main()