import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    first_line = data[0].split()
    n = int(first_line[0])
    m = int(first_line[1])
    
    A = list(map(int, data[1].split()))
    
    S = []
    for i in range(2, 2 + n):
        S.append(data[i].strip())
    
    totals = [0] * n
    for idx in range(n):
        s = S[idx]
        solved_sum = 0
        for j in range(m):
            if s[j] == 'o':
                solved_sum += A[j]
        totals[idx] = solved_sum + (idx + 1)
    
    results = []
    for i in range(n):
        max_other = -10**18
        for j in range(n):
            if j == i:
                continue
            if totals[j] > max_other:
                max_other = totals[j]
                
        if totals[i] > max_other:
            results.append(0)
        else:
            deficit = max_other - totals[i] + 1
            unsolved_scores = []
            s_str = S[i]
            for j in range(m):
                if s_str[j] == 'x':
                    unsolved_scores.append(A[j])
                    
            unsolved_scores.sort(reverse=True)
            current_sum = 0
            count = 0
            for score in unsolved_scores:
                current_sum += score
                count += 1
                if current_sum >= deficit:
                    results.append(count)
                    break
            else:
                results.append(count)
                
    for res in results:
        print(res)

if __name__ == "__main__":
    main()