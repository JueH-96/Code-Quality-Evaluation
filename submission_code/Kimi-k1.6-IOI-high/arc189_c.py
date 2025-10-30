def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    X = int(input[ptr])
    ptr += 1
    
    A = list(map(int, input[ptr:ptr+N]))
    ptr += N
    B = list(map(int, input[ptr:ptr+N]))
    ptr += N
    P = list(map(int, input[ptr:ptr+N]))
    ptr += N
    Q = list(map(int, input[ptr:ptr+N]))
    ptr += N
    
    def get_cycle(permutation, X_val):
        cycle = []
        current = X_val
        cycle.append(current)
        while True:
            next_box = permutation[current - 1]
            if next_box == X_val:
                break
            cycle.append(next_box)
            current = next_box
        return cycle
    
    cycle_P = get_cycle(P, X)
    pos_in_P = {box: idx for idx, box in enumerate(cycle_P)}
    
    cycle_Q = get_cycle(Q, X)
    pos_in_Q = {box: idx for idx, box in enumerate(cycle_Q)}
    
    s = set()
    
    # Process red balls
    for i in range(N):
        if A[i] == 1:
            box = i + 1
            if box == X:
                continue
            if box not in pos_in_P:
                print(-1)
                return
            start_idx = pos_in_P[box]
            s.update(cycle_P[start_idx:])
    
    # Process blue balls
    for i in range(N):
        if B[i] == 1:
            box = i + 1
            if box == X:
                continue
            if box not in pos_in_Q:
                print(-1)
                return
            start_idx = pos_in_Q[box]
            s.update(cycle_Q[start_idx:])
    
    print(len(s))
    
if __name__ == "__main__":
    main()