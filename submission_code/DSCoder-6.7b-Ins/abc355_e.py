import sys

def read_input():
    N, L, R = map(int, sys.stdin.readline().split())
    return N, L, R

def ask_question(i, j):
    print(f"? {i} {j}")
    sys.stdout.flush()
    return int(sys.stdin.readline().strip())

def solve():
    N, L, R = read_input()

    left, right = 0, 2**N - 1
    while left < right:
        mid = (left + right) // 2
        i, j = 0, 0
        while 2**i * (j + 1) <= mid:
            i += 1
        j = (mid // (2**i)) - 1
        res = ask_question(i, j)
        if res == -1:
            return
        if mid < L:
            left = mid + 1
        elif mid > R:
            right = mid - 1
        else:
            return res

    print(f"! {solve()}")

if __name__ == "__main__":
    solve()