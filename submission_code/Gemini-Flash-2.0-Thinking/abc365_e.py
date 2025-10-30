def solve():
    n = int(input())
    a = list(map(int, input().split()))

    total_xor_sum = 0
    for i in range(n - 1):
        current_xor = 0
        for j in range(i, n):
            current_xor ^= a[j]
            if j > i:
                total_xor_sum += (current_xor ^ a[j])

    print(total_xor_sum)

def solve_optimized():
    n = int(input())
    a = list(map(int, input().split()))

    prefix_xor = [0] * (n + 1)
    for i in range(n):
        prefix_xor[i + 1] = prefix_xor[i] ^ a[i]

    total_xor_sum = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            subarray_xor = prefix_xor[j + 1] ^ prefix_xor[i]
            total_xor_sum += subarray_xor
    print(total_xor_sum)

if __name__ == "__main__":
    solve_optimized()