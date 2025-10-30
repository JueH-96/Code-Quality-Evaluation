def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n, m = map(int, data[0].split())
    A = list(map(int, data[1].split()))
    S_list = []
    for i in range(2, 2+n):
        S_list.append(data[i].strip())
    
    total_scores = []
    for i in range(n):
        s = S_list[i]
        base = 0
        for j in range(m):
            if s[j] == 'o':
                base += A[j]
        total_scores.append(base + (i+1))
    
    results = []
    for i in range(n):
        max_other = 0
        for j in range(n):
            if j == i:
                continue
            if total_scores[j] > max_other:
                max_other = total_scores[j]
                
        current_i = total_scores[i]
        if current_i > max_other:
            results.append(0)
            continue
            
        required = max_other + 1 - current_i
        unsolved = []
        s_i = S_list[i]
        for idx in range(m):
            if s_i[idx] == 'x':
                unsolved.append(A[idx])
                
        unsolved.sort(reverse=True)
        s = 0
        count = 0
        for score in unsolved:
            s += score
            count += 1
            if s >= required:
                break
        results.append(count)
    
    for res in results:
        print(res)

if __name__ == "__main__":
    main()