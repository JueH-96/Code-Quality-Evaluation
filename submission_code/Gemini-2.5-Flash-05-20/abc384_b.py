# YOUR CODE HERE
import sys

def solve():
    N, R = map(int, sys.stdin.readline().split())

    current_rating = R

    for _ in range(N):
        D, A = map(int, sys.stdin.readline().split())

        is_subject_to_update = False
        if D == 1:  # ARC Div. 1
            if 1600 <= current_rating <= 2799:
                is_subject_to_update = True
        elif D == 2:  # ARC Div. 2
            if 1200 <= current_rating <= 2399:
                is_subject_to_update = True
        
        if is_subject_to_update:
            current_rating += A
            
    print(current_rating)

if __name__ == '__main__':
    solve()