import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    Q = int(data[idx])
    idx +=1
    
    history = [(1, 0)]
    for _ in range(Q):
        query = data[idx]
        idx +=1
        if query == '1':
            c = data[idx]
            idx +=1
            x, y = history[-1]
            if c == 'U':
                y +=1
            elif c == 'D':
                y -=1
            elif c == 'R':
                x +=1
            elif c == 'L':
                x -=1
            history.append((x, y))
        else:
            p = int(data[idx])
            idx +=1
            m = len(history) -1
            if m >= (p-1):
                target = m - (p-1)
                x, y = history[target]
            else:
                x = p - m
                y = 0
            print(f"{x} {y}")

if __name__ == '__main__':
    main()