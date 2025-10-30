def solve():
    n = int(input())
    a = list(map(int, input().split()))
    
    counts = [0] * (n + 1)
    counts[1] = n
    counts[2] = n * (n - 1) // 2
    
    ans_counts = [0] * (n + 1)
    mod = 998244353
    
    def get_arithmetic_subsequences(last_index, current_length, common_difference):
        if current_length >= 3:
            ans_counts[current_length] = (ans_counts[current_length] + 1) % mod
        if current_length == n:
            return
        for i in range(last_index + 1, n):
            if a[i] == a[last_index] + common_difference:
                get_arithmetic_subsequences(i, current_length + 1, common_difference)
                
    for i1 in range(n):
        for i2 in range(i1 + 1, n):
            common_diff = a[i2] - a[i1]
            get_arithmetic_subsequences(i2, 2, common_diff)
            
    result = []
    result.append(str(counts[1]))
    result.append(str(counts[2] % mod))
    for k in range(3, n + 1):
        result.append(str(ans_counts[k]))
        
    print(*(result))

if __name__ == '__main__':
    solve()