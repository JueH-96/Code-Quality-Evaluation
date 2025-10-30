def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    R = int(input[1])
    C = int(input[2])
    S = input[3]
    
    current_x = 0
    current_y = 0
    seen = set()
    seen.add((current_x, current_y))
    res = []
    
    for c in S:
        dx, dy = 0, 0
        if c == 'N':
            dx = -1
        elif c == 'S':
            dx = 1
        elif c == 'W':
            dy = -1
        elif c == 'E':
            dy = 1
        
        new_x = current_x + dx
        new_y = current_y + dy
        
        target_x = new_x - R
        target_y = new_y - C
        
        if (target_x, target_y) in seen:
            res.append('1')
        else:
            res.append('0')
        
        current_x, current_y = new_x, new_y
        seen.add((current_x, current_y))
    
    print(''.join(res))

if __name__ == "__main__":
    main()