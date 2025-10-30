# YOUR CODE HERE
def solve():
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    
    mod = 998244353
    
    count = 0
    for i in range(1 << (n - 1)):
        subsequences = []
        current_subsequence = []
        for j in range(n):
            current_subsequence.append(a[j])
            if j < n - 1 and (i >> j) & 1:
                subsequences.append(current_subsequence)
                current_subsequence = []
        subsequences.append(current_subsequence)
        
        valid = True
        for subsequence in subsequences:
            if sum(subsequence) == k:
                valid = False
                break
        
        if valid:
            count = (count + 1) % mod
            
    print(count)

solve()