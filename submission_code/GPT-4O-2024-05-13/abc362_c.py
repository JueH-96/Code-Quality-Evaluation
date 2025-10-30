# YOUR CODE HERE
import sys
input = sys.stdin.read

def solve():
    data = input().strip().split()
    N = int(data[0])
    intervals = [(int(data[2*i+1]), int(data[2*i+2])) for i in range(N)]
    
    min_sum = sum(L for L, R in intervals)
    max_sum = sum(R for L, R in intervals)
    
    if min_sum > 0 or max_sum < 0:
        print("No")
        return
    
    target = 0
    current_sum = min_sum
    X = [L for L, R in intervals]
    
    for i in range(N):
        if current_sum == target:
            break
        L, R = intervals[i]
        if current_sum < target:
            diff = min(R - L, target - current_sum)
            X[i] += diff
            current_sum += diff
        elif current_sum > target:
            diff = min(X[i] - L, current_sum - target)
            X[i] -= diff
            current_sum -= diff
    
    print("Yes")
    print(" ".join(map(str, X)))

solve()