# YOUR CODE HERE
import sys
import collections

def solve():
    import sys
    from collections import deque
    input = sys.stdin.read().split()
    Q = int(input[0])
    tokens = input[1:]
    dq = deque()
    current_day = 0
    res = []
    i =0
    for _ in range(Q):
        cmd = tokens[i]
        if cmd == '1':
            if dq and dq[-1][0] == current_day:
                dq[-1][1] +=1
            else:
                dq.append([current_day,1])
            i +=1
        elif cmd == '2':
            T = int(tokens[i+1])
            current_day += T
            i +=2
        elif cmd == '3':
            H = int(tokens[i+1])
            thresh = current_day - H
            harvested =0
            while dq and dq[0][0] <= thresh:
                harvested += dq[0][1]
                dq.popleft()
            res.append(str(harvested))
            i +=2
    print('
'.join(res))
    
if __name__ == "__main__":
    solve()