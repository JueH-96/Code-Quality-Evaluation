import sys

def main():
    n, m = map(int, sys.stdin.readline().split())
    a = list(map(int, sys.stdin.readline().split()))
    s = [sys.stdin.readline().strip() for _ in range(n)]
    
    current_scores = []
    for i in range(n):
        total = 0
        for j in range(m):
            if s[i][j] == 'o':
                total += a[j]
        total += (i + 1)
        current_scores.append(total)
    
    for i in range(n):
        max_other = max(current_scores[j] for j in range(n) if j != i)
        current = current_scores[i]
        if current > max_other:
            print(0)
        else:
            required = max_other - current + 1
            unsolved = [a[j] for j in range(m) if s[i][j] == 'x']
            unsolved.sort(reverse=True)
            sum_so_far = 0
            count = 0
            for val in unsolved:
                sum_so_far += val
                count += 1
                if sum_so_far >= required:
                    break
            print(count)

if __name__ == "__main__":
    main()