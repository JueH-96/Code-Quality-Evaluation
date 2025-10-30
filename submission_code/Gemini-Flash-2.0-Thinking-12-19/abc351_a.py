import sys

def solve():
    line1 = sys.stdin.readline().split()
    a = [int(x) for x in line1]
    line2 = sys.stdin.readline().split()
    b = [int(x) for x in line2]
    
    team_takahashi_score = sum(a)
    team_aoki_score = sum(b)
    
    runs_needed = (team_takahashi_score - team_aoki_score) + 1
    print(runs_needed)

if __name__ == '__main__':
    solve()