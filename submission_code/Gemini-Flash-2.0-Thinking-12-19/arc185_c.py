def solve():
    n, x = map(int, input().split())
    a = list(map(int, input().split()))
    
    for i in range(n):
        seen_values = {}
        for j in range(i + 1, n):
            complement = x - a[i] - a[j]
            if complement in seen_values:
                k_index = j
                j_index = seen_values[complement]
                if j_index > i and k_index > j_index:
                    print(i + 1, j_index + 1, k_index + 1)
                    return
            seen_values[a[j]] = j
            
    print("-1")

if __name__ == '__main__':
    solve()