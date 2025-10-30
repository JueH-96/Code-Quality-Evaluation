import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    results = []
    for _ in range(T):
        N = int(input[idx])
        idx += 1
        P = list(map(int, input[idx:idx+N]))
        idx += N
        
        max_up_to = [0] * N
        max_up_to[0] = P[0]
        for i in range(1, N):
            max_up_to[i] = max(max_up_to[i-1], P[i])
        
        breakpoints = 0
        for i in range(N):
            if max_up_to[i] > i + 1:  # positions are 1-based in the problem
                breakpoints += 1
        
        # Subtract the number of consecutive breakpoints
        consecutive = 0
        prev_broken = False
        for i in range(N):
            current_broken = max_up_to[i] > (i + 1)
            if current_broken:
                if prev_broken:
                    consecutive += 1
                prev_broken = True
            else:
                prev_broken = False
        results.append(max(0, breakpoints - consecutive))
    
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()