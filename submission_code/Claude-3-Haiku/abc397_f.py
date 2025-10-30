from collections import defaultdict

def count_distinct(arr):
    return len(set(arr))

def solve():
    N = int(input())
    A = list(map(int, input().split()))
    
    max_sum = 0
    
    for i in range(1, N):
        for j in range(i+1, N):
            left = A[:i]
            middle = A[i:j]
            right = A[j:]
            
            left_distinct = count_distinct(left)
            middle_distinct = count_distinct(middle)
            right_distinct = count_distinct(right)
            
            max_sum = max(max_sum, left_distinct + middle_distinct + right_distinct)
    
    print(max_sum)

solve()