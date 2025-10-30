def main():
    import sys
    data = sys.stdin.read().splitlines()
    n = int(data[0].strip())
    s = data[1].strip()
    
    if n == 0:
        print(0)
        return
        
    moves_map = {
        'R': ['R', 'P'],
        'P': ['P', 'S'],
        'S': ['S', 'R']
    }
    
    prev0 = 0
    prev1 = 1
    
    if n == 1:
        print(max(prev0, prev1))
        return
        
    for i in range(1, n):
        c = s[i]
        prev_c = s[i-1]
        curr_moves = moves_map[c]
        prev_moves = moves_map[prev_c]
        
        cand0 = []
        if prev_moves[0] != curr_moves[0]:
            cand0.append(prev0)
        if prev_moves[1] != curr_moves[0]:
            cand0.append(prev1)
        curr0 = max(cand0) if cand0 else -10**18
        
        cand1 = []
        if prev_moves[0] != curr_moves[1]:
            cand1.append(prev0)
        if prev_moves[1] != curr_moves[1]:
            cand1.append(prev1)
        curr1 = max(cand1) + 1 if cand1 else -10**18
        
        prev0, prev1 = curr0, curr1
        
    print(max(prev0, prev1))

if __name__ == "__main__":
    main()