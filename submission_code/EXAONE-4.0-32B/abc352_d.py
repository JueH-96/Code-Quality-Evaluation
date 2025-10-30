import sys
from collections import deque

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    p = list(map(int, data[2:2+n]))
    
    pos = [0] * (n + 1)
    for idx, num in enumerate(p):
        pos[num] = idx
        
    B = [0] * n
    for i in range(1, n + 1):
        B[i - 1] = pos[i]
        
    min_deq = deque()
    max_deq = deque()
    ans = float('inf')
    
    for i in range(k):
        while min_deq and B[min_deq[-1]] > B[i]:
            min_deq.pop()
        min_deq.append(i)
        
        while max_deq and B[max_deq[-1]] < B[i]:
            max_deq.pop()
        max_deq.append(i)
        
    current_min = B[min_deq[0]]
    current_max = B[max_deq[0]]
    ans = current_max - current_min
    
    for l in range(1, n - k + 1):
        r = l + k - 1
        
        if min_deq and min_deq[0] == l - 1:
            min_deq.popleft()
        if max_deq and max_deq[0] == l - 1:
            max_deq.popleft()
            
        while min_deq and B[min_deq[-1]] > B[r]:
            min_deq.pop()
        min_deq.append(r)
        
        while max_deq and B[max_deq[-1]] < B[r]:
            max_deq.pop()
        max_deq.append(r)
        
        current_min = B[min_deq[0]]
        current_max = B[max_deq[0]]
        diff = current_max - current_min
        if diff < ans:
            ans = diff
            
    print(ans)

if __name__ == '__main__':
    main()