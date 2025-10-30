import sys
import threading

def main():
    import sys
    data = sys.stdin
    n_line = data.readline().strip()
    if not n_line:
        return
    N = int(n_line)
    total_A = 0
    max_diff = 0
    for _ in range(N):
        a_str = data.readline().split()
        A = int(a_str[0])
        B = int(a_str[1])
        total_A += A
        diff = B - A
        if diff > max_diff:
            max_diff = diff
    # The maximum head height is sum of all A_i plus the maximum (B_i - A_i)
    print(total_A + max_diff)

if __name__ == "__main__":
    main()