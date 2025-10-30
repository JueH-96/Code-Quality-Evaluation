def solve():
    n, l, r = map(int, input().split())
    total_sum = 0
    current_l = l
    current_r = r
    i = n
    queries = []
    while current_l <= current_r:
        best_i = -1
        for power in range(n, -1, -1):
            block_size = 1 << power
            if current_l % block_size == 0 and current_l + block_size - 1 <= current_r and 2**power <= (1 << n):
                best_i = power
                break
        if best_i == -1:
            best_i = 0
            block_size = 1
            if not (current_l % block_size == 0 and current_l + block_size - 1 <= current_r and 2**0 <= (1 << n)):
                break

        block_size = 1 << best_i
        j = current_l // block_size
        queries.append((best_i, j))
        current_l += block_size

    for i, j in queries:
        print(f"? {i} {j}")
        response = int(input())
        if response == -1:
            return
        total_sum = (total_sum + response) % 100

    print(f"! {total_sum}")

if __name__ == '__main__':
    solve()