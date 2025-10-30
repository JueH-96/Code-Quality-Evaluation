from itertools import accumulate

def solve(N, A):
    # 各位置i的最左大值和最右大值
    max_lefts = list(accumulate(A, max))
    max_rights = list(accumulate(A[::-1], max))[::-1]
    
    # 跳跃数组，用于快速查找左侧或右侧第一个较大值的位置
    jumps_right = [N - 1] * N
    jumps_left = [0] * N
    
    stack = []
    for i, size in enumerate(A):
        while stack and A[stack[-1]] <= size:
            idx = stack.pop()
            jumps_left[idx] = i
        if stack:
            jumps_right[i] = stack[-1]
        stack.append(i)
    
    # 能获得的最大大小
    result = A.copy()
    for i in range(1, N):
        # 向左向右尝试吸收的最大的值
        best_left = A[i] + result[jumps_left[i]] if jumps_left[i] < i else 0
        best_right = A[i] + result[jumps_right[i]] if jumps_right[i] > i else 0
        
        # 将左边最大值或右边最大值，和max_lefts和max_rights中取最大
        result[i] = max(best_left, best_right, max_lefts[i], max_rights[i])
    
    return result

N = int(input())
A = list(map(int, input().split()))
result = solve(N, A)
print(*result)