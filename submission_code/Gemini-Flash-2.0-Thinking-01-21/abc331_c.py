def solve():
    n = int(input())
    a = list(map(int, input().split()))
    max_val = 1000000
    frequency = [0] * (max_val + 1)
    for x in a:
        frequency[x] += 1
    
    suffix_sum = [0] * (max_val + 2)
    for v in range(max_val, 0, -1):
        suffix_sum[v] = v * frequency[v] + suffix_sum[v+1]
        
    results = []
    for i in range(n):
        current_a_i = a[i]
        if current_a_i >= max_val:
            results.append(0)
        else:
            results.append(suffix_sum[current_a_i + 1])
            
    print(*(results))

if __name__ == '__main__':
    solve()