import sys

def solve():
    n, m = map(int, sys.stdin.readline().split())
    h_list = list(map(int, sys.stdin.readline().split()))
    
    remaining_disinfectant = m
    success_count = 0
    
    for hands_needed in h_list:
        if remaining_disinfectant >= hands_needed:
            remaining_disinfectant -= hands_needed
            success_count += 1
        else:
            remaining_disinfectant = 0
            
    print(success_count)

if __name__ == '__main__':
    solve()