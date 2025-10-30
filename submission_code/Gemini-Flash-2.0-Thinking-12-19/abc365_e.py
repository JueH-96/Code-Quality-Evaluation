def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    prefix_xor_sums = [0] * (n + 1)
    for i in range(n):
        prefix_xor_sums[i+1] = prefix_xor_sums[i] ^ a[i]
        
    total_xor_sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            current_xor_sum = 0
            for k in range(i, j + 1):
                current_xor_sum ^= a[k]
            total_xor_sum += current_xor_sum
            
    print(total_xor_sum)

if __name__ == '__main__':
    solve()