# YOUR CODE HERE
import sys
import math

def is_same_tile(S_x, S_y, T_x, T_y):
    def get_tile(x, y):
        i = int(x)
        j = int(y)
        if (i + j) % 2 == 0:
            # Horizontal tile covering x in [i, i+2], y in [j, j+1]
            return ('h', i // 2, j)
        else:
            # Vertical tile covering x in [i, i+1], y in [j, j+2]
            return ('v', i, j // 2)
    return get_tile(S_x + 0.5, S_y + 0.5) == get_tile(T_x + 0.5, T_y + 0.5)

def main():
    import sys
    input = sys.stdin.readline
    S_x_str, S_y_str = sys.stdin.readline().split()
    T_x_str, T_y_str = sys.stdin.readline().split()
    S_x = int(S_x_str)
    S_y = int(S_y_str)
    T_x = int(T_x_str)
    T_y = int(T_y_str)
    if is_same_tile(S_x, S_y, T_x, T_y):
        print(0)
        return
    delta_x = abs(S_x - T_x)
    delta_y = abs(S_y - T_y)
    if (delta_x == 1 and delta_y == 0) or (delta_x == 0 and delta_y == 1):
        # Special case where toll can be zero
        if is_same_tile((S_x + T_x) // 2, (S_y + T_y) // 2, S_x, S_y):
            print(0)
            return
    toll = (delta_x + 1) // 2 + (delta_y + 1) // 2
    print(toll)

if __name__ == "__main__":
    main()