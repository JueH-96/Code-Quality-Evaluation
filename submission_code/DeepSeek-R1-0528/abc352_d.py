import collections

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    k = int(data[1])
    P = list(map(int, data[2:2+n]))
    
    pos = [0] * (n + 1)
    for idx in range(n):
        x = P[idx]
        pos[x] = idx
        
    B = [0] * n
    for i in range(1, n + 1):
        B[i - 1] = pos[i]
    
    min_deque = collections.deque()
    max_deque = collections.deque()
    min_arr = []
    max_arr = []
    
    for i in range(n):
        while min_deque and B[min_deque[-1]] > B[i]:
            min_deque.pop()
        min_deque.append(i)
        
        while max_deque and B[max_deque[-1]] < B[i]:
            max_deque.pop()
        max_deque.append(i)
        
        if i >= k - 1:
            min_arr.append(B[min_deque[0]])
            max_arr.append(B[max_deque[0]])
            
            if min_deque[0] == i - k + 1:
                min_deque.popleft()
            if max_deque[0] == i - k + 1:
                max_deque.popleft()
                
    ans = 10**18
    for i in range(len(min_arr)):
        diff = max_arr[i] - min_arr[i]
        if diff < ans:
            ans = diff
            
    print(ans)

if __name__ == '__main__':
    main()