import sys

def query(i, j):
    print(f"? {i} {j}")
    sys.stdout.flush()
    return int(input())

def solve(N, L, R):
    def get_sum(l, r):
        if l == r:
            return query(0, l)
        i = 0
        while 2 ** (i + 1) <= r - l + 1:
            i += 1
        j = l // (2 ** i)
        if j * (2 ** i) == l and (j + 1) * (2 ** i) - 1 == r:
            return query(i, j)
        else:
            return (get_sum(l, (j + 1) * (2 ** i) - 1) + get_sum((j + 1) * (2 ** i), r)) % 100

    return get_sum(L, R)

def main():
    N, L, R = map(int, input().split())
    result = solve(N, L, R)
    print(f"! {result}")

if __name__ == "__main__":
    main()