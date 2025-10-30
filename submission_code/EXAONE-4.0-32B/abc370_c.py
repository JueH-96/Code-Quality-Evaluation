def main():
    S = input().strip()
    T = input().strip()
    
    n = len(S)
    diff = set()
    for i in range(n):
        if S[i] != T[i]:
            diff.add(i)
            
    M = len(diff)
    current = S
    steps = []
    
    for _ in range(M):
        best_next = None
        best_j = None
        for j in diff:
            candidate = current[:j] + T[j] + current[j+1:]
            if best_next is None or candidate < best_next:
                best_next = candidate
                best_j = j
                
        steps.append(best_next)
        current = best_next
        diff.remove(best_j)
        
    print(M)
    for s in steps:
        print(s)

if __name__ == "__main__":
    main()