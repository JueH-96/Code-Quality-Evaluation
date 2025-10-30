def solve():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))

    count = 0
    mod = 998244353

    def check(x):
        for i in range(n):
            if x[i] > x[a[i]-1]:
                return False
        return True

    def generate_sequences(index, current_sequence):
        nonlocal count
        if index == n:
            if check(current_sequence):
                count = (count + 1) % mod
            return

        for i in range(1, m + 1):
            generate_sequences(index + 1, current_sequence + [i])

    generate_sequences(0, [])
    print(count)

solve()