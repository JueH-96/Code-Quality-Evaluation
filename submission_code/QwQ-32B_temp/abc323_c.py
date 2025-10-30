def main():
    import sys
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    S_list = [sys.stdin.readline().strip() for _ in range(N)]
    
    current_scores = []
    for i in range(N):
        total = 0
        s = S_list[i]
        for j in range(M):
            if s[j] == 'o':
                total += A[j]
        total += (i + 1)
        current_scores.append(total)
    
    for i in range(N):
        current_i = current_scores[i]
        others = []
        for j in range(N):
            if j != i:
                others.append(current_scores[j])
        max_others = max(others)
        if current_i > max_others:
            print(0)
            continue
        required = max_others - current_i + 1
        unsolved = []
        for j in range(M):
            if S_list[i][j] == 'x':
                unsolved.append(A[j])
        unsolved_sorted = sorted(unsolved, reverse=True)
        sum_needed = 0
        count = 0
        for a in unsolved_sorted:
            sum_needed += a
            count += 1
            if sum_needed >= required:
                break
        print(count)
    
if __name__ == "__main__":
    main()