# YOUR CODE HERE
from bisect import bisect_left, bisect_right

def inversion_count(arr):
    sorted_arr = []
    count = 0
    for num in arr:
        i = bisect_right(sorted_arr, num)
        count += i
        bisect_left(sorted_arr, num)
    return count

def solve(N, P, M, A):
    inv_count = 0
    sorted_P = []
    result = []
    for i in range(N):
        sorted_P.append(P[i])
    
    for i in range(M):
        k = A[i]
        for j in range(k-1):
            if P[j] > P[j+1]:
                P[j], P[j+1] = P[j+1], P[j]
                inv_count += j + 1 - bisect_left(sorted_P, P[j])
                inv_count -= bisect_right(sorted_P, P[j+1])
                sorted_P.remove(P[j])
                sorted_P.insert(bisect_left(sorted_P, P[j]), P[j])
                sorted_P.remove(P[j+1])
                sorted_P.insert(bisect_right(sorted_P, P[j+1]), P[j+1])
        result.append(inv_count)
    
    for r in result:
        print(r)