import sys

def main():
    N, L, R = map(int, sys.stdin.readline().split())
    total = 0
    left = L
    right_end = R
    while left <= right_end:
        best_i = 0
        # Find the best i from highest to lowest
        for i in range(N, -1, -1):
            block_size = 1 << i
            if (left % block_size) != 0:
                continue
            if (left + block_size - 1) > right_end:
                continue
            best_i = i
            break
        j = left // (1 << best_i)
        print(f"? {best_i} {j}")
        sys.stdout.flush()
        t = int(sys.stdin.readline())
        total = (total + t) % 100
        left += (1 << best_i)
    print(f"! {total % 100}")

if __name__ == "__main__":
    main()