def solve():
    n = int(input())
    a = list(map(int, input().split()))
    prefix_xor = [0] * (n + 1)
    for i in range(n):
        prefix_xor[i+1] = prefix_xor[i] ^ a[i]
    
    total_xor_sum = 0
    for i in range(1, n):
        for j in range(i + 1, n + 1):
            current_xor_sum = prefix_xor[i-1] ^ prefix_xor[j]
            total_xor_sum += current_xor_sum
            
    print(total_xor_sum)

if __name__ == '__main__':
    solve()