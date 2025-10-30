def main():
    S = input().strip()
    T = input().strip()
    n = len(S)
    available = set()
    for i in range(n):
        if S[i] != T[i]:
            available.add(i)
            
    d = len(available)
    if d == 0:
        print(0)
        return
        
    base = S
    steps = []
    for k in range(d):
        best_candidate = None
        best_index = None
        for i in available:
            candidate_str = base[:i] + T[i] + base[i+1:]
            if best_candidate is None or candidate_str < best_candidate:
                best_candidate = candidate_str
                best_index = i
                
        steps.append(best_candidate)
        base = best_candidate
        available.remove(best_index)
        
    print(d)
    for s in steps:
        print(s)

if __name__ == "__main__":
    main()