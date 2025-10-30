import sys

def solve():
    n_l_line = sys.stdin.readline().split()
    n = int(n_l_line[0])
    l = int(n_l_line[1])
    scores_line = sys.stdin.readline().split()
    scores = [int(score) for score in scores_line]
    
    pass_count = 0
    for score in scores:
        if score >= l:
            pass_count += 1
            
    print(pass_count)

if __name__ == '__main__':
    solve()