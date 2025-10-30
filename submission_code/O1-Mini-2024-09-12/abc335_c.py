# YOUR CODE HERE
import sys

def main():
    import sys
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    queries = data[2:]
    
    head_x = [1]
    head_y = [0]
    t = 0  # Number of moves made
    res = []
    idx = 0
    for _ in range(Q):
        cmd = queries[idx]
        if cmd == '1':
            C = queries[idx +1]
            last_x = head_x[-1]
            last_y = head_y[-1]
            if C == 'R':
                new_x = last_x +1
                new_y = last_y
            elif C == 'L':
                new_x = last_x -1
                new_y = last_y
            elif C == 'U':
                new_x = last_x
                new_y = last_y +1
            elif C == 'D':
                new_x = last_x
                new_y = last_y -1
            head_x.append(new_x)
            head_y.append(new_y)
            t +=1
            idx +=2
        elif cmd == '2':
            p = int(queries[idx +1])
            if t >= p -1:
                move_idx = t - (p -1)
                x = head_x[move_idx]
                y = head_y[move_idx]
            else:
                x = p - t
                y = 0
            res.append(f"{x} {y}")
            idx +=2
    print('
'.join(res))

if __name__ == "__main__":
    main()