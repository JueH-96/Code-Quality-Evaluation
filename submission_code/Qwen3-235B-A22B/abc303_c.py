import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx +=1
    M = int(input[idx]); idx +=1
    H = int(input[idx]); idx +=1
    K = int(input[idx]); idx +=1
    S = input[idx]; idx +=1
    
    items = set()
    for _ in range(M):
        x = int(input[idx])
        y = int(input[idx+1])
        items.add((x, y))
        idx +=2
    
    x, y = 0, 0
    current_h = H
    
    for c in S:
        if c == 'R':
            x +=1
        elif c == 'L':
            x -=1
        elif c == 'U':
            y +=1
        elif c == 'D':
            y -=1
        
        current_h -=1
        if current_h < 0:
            print("No")
            return
        
        pos = (x, y)
        if pos in items and current_h < K:
            items.remove(pos)
            current_h = K
    
    print("Yes")

if __name__ == "__main__":
    main()