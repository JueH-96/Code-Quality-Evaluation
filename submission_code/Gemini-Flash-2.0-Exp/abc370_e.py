def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    count = 0
    
    for i in range(2**(n-1)):
        divisions = []
        current_subsequence = []
        
        for j in range(n):
            current_subsequence.append(a[j])
            if j < n - 1 and (i >> j) & 1:
                divisions.append(current_subsequence)
                current_subsequence = []
        divisions.append(current_subsequence)
        
        valid = True
        for subsequence in divisions:
            if sum(subsequence) == k:
                valid = False
                break
        
        if valid:
            count = (count + 1) % 998244353
            
    print(count)

solve()