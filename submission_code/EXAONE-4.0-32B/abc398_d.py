import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    parts = data[0].split()
    N = int(parts[0])
    R = int(parts[1])
    C = int(parts[2])
    S = data[1].strip()
    
    x, y = 0, 0
    first_occurrence = {}
    first_occurrence[(0, 0)] = 0
    results = []
    
    for i in range(N):
        char = S[i]
        if char == 'N':
            x -= 1
        elif char == 'S':
            x += 1
        elif char == 'W':
            y -= 1
        elif char == 'E':
            y += 1
        
        if (x, y) == (R, C):
            results.append('1')
        else:
            qx = x - R
            qy = y - C
            Q = (qx, qy)
            if Q in first_occurrence:
                k0 = first_occurrence[Q]
                if k0 >= 1:
                    results.append('1')
                else:
                    results.append('0')
            else:
                results.append('0')
                
        if (x, y) not in first_occurrence:
            first_occurrence[(x, y)] = i + 1
            
    print(''.join(results))

if __name__ == "__main__":
    main()