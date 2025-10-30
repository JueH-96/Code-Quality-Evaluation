def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    prefix_xor = [0] * (N + 1)
    for i in range(1, N + 1):
        prefix_xor[i] = prefix_xor[i - 1] ^ A[i - 1]
    
    sum_over_all_subarrays = 0
    for k in range(30):
        count = {0: 1, 1: 0}
        S_k = 0
        for j in range(1, N + 1):
            b = (prefix_xor[j] >> k) & 1
            S_k += count[1 - b]
            count[b] += 1
        sum_over_all_subarrays += S_k * (1 << k)
    
    sum_A = sum(A)
    sum_over_subarrays_of_length_at_least2 = sum_over_all_subarrays - sum_A
    print(sum_over_subarrays_of_length_at_least2)

if __name__ == "__main__":
    main()