def query(i, j):
    print(f"? {i} {j}")
    return int(input())

def main():
    N, L, R = map(int, input().split())
    sum_remainder = 0
    for i in range(N-1, -1, -1):
        if L & (1 << i) != 0:
            sum_remainder += query(i, L >> i)
            L += 1 << i
        if R & (1 << i) == 0:
            sum_remainder += query(i, (R >> i) - 1)
            R -= 1 << i
        sum_remainder %= 100
    if L <= R:
        sum_remainder += query(0, L)
        sum_remainder %= 100
    print(f"! {sum_remainder}")

if __name__ == "__main__":
    main()