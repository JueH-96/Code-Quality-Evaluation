import sys

def main():
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    total = 0

    for i in range(N):
        a = A[i]
        b = B[i]
        d_clockwise = (b - a) % M
        d_counter = (a - b) % M
        valid_clockwise = True
        valid_counter = True

        if i > 0:
            B_prev = B[i-1]
            if (B_prev - a) % M <= d_clockwise:
                valid_clockwise = False
            if (a - B_prev) % M <= d_counter:
                valid_counter = False

        if i < N - 1:
            B_next = B[i+1]
            if (B_next - a) % M <= d_clockwise:
                valid_clockwise = False
            if (a - B_next) % M <= d_counter:
                valid_counter = False

        if not valid_clockwise and not valid_counter:
            print(-1)
            return

        if valid_clockwise and valid_counter:
            if d_clockwise <= d_counter:
                total += d_clockwise
            else:
                total += d_counter
        elif valid_clockwise:
            total += d_clockwise
        else:
            total += d_counter

    print(total)

if __name__ == "__main__":
    main()