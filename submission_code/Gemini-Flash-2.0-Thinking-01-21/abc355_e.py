def solve():
    n, l, r = map(int, input().split())
    total_sum = 0
    current_l = l
    current_r = r
    while current_l <= current_r:
        best_i = 0
        for i in range(n - 1, -1, -1):
            if current_l % (2**i) == 0 and current_l + (2**i) - 1 <= current_r:
                best_i = i
                break
        j = current_l // (2**best_i)
        print(f"? {best_i} {j}")
        response = int(input())
        if response == -1:
            return
        total_sum = (total_sum + response) % 100
        current_l += 2**best_i
    print(f"! {total_sum}")

if __name__ == '__main__':
    solve()