def solve():
    n, l, r = map(int, input().split())

    total_sum = 0
    current_l = l
    current_r = r

    while current_l <= current_r:
        best_i = -1
        best_block_size = -1

        for i in range(n + 1):
            block_size = 2**i
            if current_l % block_size == 0 and current_l + block_size - 1 <= current_r:
                best_i = i
                best_block_size = block_size
            else:
                break

        if best_block_size > 0:
            j = current_l // best_block_size
            print(f"? {best_i} {j}")
            response = int(input())
            if response == -1:
                return
            total_sum = (total_sum + response) % 100
            current_l += best_block_size
        else:
            print(f"? 0 {current_l}")
            response = int(input())
            if response == -1:
                return
            total_sum = (total_sum + response) % 100
            current_l += 1

    print(f"! {total_sum}")

if __name__ == "__main__":
    solve()