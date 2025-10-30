# N = int(input())
# A = list(map(int, input().split()))
# S = input()

def solve(N, A, S):
    from collections import defaultdict
    
    def mex(arr):
        mex_val = 0
        while mex_val in arr:
            mex_val += 1
        return mex_val
    
    result = 0
    for i in range(N-2):
        if S[i] == 'M':
            for j in range(i+1, N-1):
                if S[j] == 'E':
                    for k in range(j+1, N):
                        if S[k] == 'X':
                            result += mex([A[i], A[j], A[k]])
    return result

N = int(input())
A = list(map(int, input().split()))
S = input()
print(solve(N, A, S))