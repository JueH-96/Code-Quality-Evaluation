from collections import deque
import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+n]))
    
    q = deque(A)
    count = 0
    current = K
    
    while q:
        if current < q[0]:
            count += 1
            current = K
        else:
            g = q.popleft()
            current -= g
            
    count += 1
    print(count)

if __name__ == '__main__':
    main()