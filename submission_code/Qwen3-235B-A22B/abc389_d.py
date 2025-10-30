import math

def main():
    R = int(input())
    total = 0
    R_squared = R * R
    for a in range(R + 1):
        x = a + 0.5
        x_squared = x * x
        s = R_squared - x_squared
        if s < 0:
            break
        if s == 0:
            max_b_plus_half = 0.0
        else:
            max_b_plus_half = math.sqrt(s)
        b_max_candidate = max_b_plus_half - 0.5
        b_max = math.floor(b_max_candidate)
        if b_max < 0:
            continue
        num_b = b_max + 1
        if a == 0:
            contrib = 2 * num_b - 1
        else:
            contrib = 4 * num_b - 2
        total += contrib
    print(total)

if __name__ == "__main__":
    main()