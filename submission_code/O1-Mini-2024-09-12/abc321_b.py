# YOUR CODE HERE
def main():
    import sys
    N_X = sys.stdin.readline().strip()
    while N_X == '':
        N_X = sys.stdin.readline().strip()
    N, X = map(int, N_X.split())
    A_line = ''
    while len(A_line.split()) < N-1:
        A_line += ' ' + sys.stdin.readline().strip()
    A = list(map(int, A_line.split()))
    
    for S in range(101):
        scores = A + [S]
        sum_all = sum(scores)
        min_score = min(scores)
        max_score = max(scores)
        final_grade = sum_all - min_score - max_score
        if final_grade >= X:
            print(S)
            return
    print(-1)

if __name__ == "__main__":
    main()