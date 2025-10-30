import bisect

def solve():
    n, m, d = map(int, input().split())
    a = list(map(int, input().split()))
    b = list(map(int, input().split()))

    max_sum = -1

    for val_a in a:
        for val_b in b:
            if abs(val_a - val_b) <= d:
                max_sum = max(max_sum, val_a + val_b)

    print(max_sum)

if __name__ == "__main__":
    solve()