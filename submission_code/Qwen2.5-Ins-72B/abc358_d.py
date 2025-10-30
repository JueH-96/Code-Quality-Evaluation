# YOUR CODE HERE
import sys

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    A = list(map(int, data[2:2+N]))
    B = list(map(int, data[2+N:]))
    
    A.sort()
    B.sort()
    
    if A[-M] < B[0]:
        print(-1)
        return
    
    left = 0
    right = 10**18
    while left < right:
        mid = (left + right) // 2
        count = 0
        for a in A:
            if a <= mid:
                count += 1
        if count >= M:
            right = mid
        else:
            left = mid + 1
    
    boxes = []
    for a in A:
        if a <= left:
            boxes.append(a)
        if len(boxes) == M:
            break
    
    boxes.sort(reverse=True)
    B.sort(reverse=True)
    
    for i in range(M):
        if boxes[i] < B[i]:
            print(-1)
            return
    
    print(sum(boxes))

solve()