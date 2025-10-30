import math

def solve():
    n = int(input())
    p = list(map(int, input().split()))
    a = list(map(int, input().split()))
    
    current_a = list(a)
    initial_a = list(a)
    best_a = list(a)
    
    while True:
        next_a = [0] * n
        for i in range(n):
            next_a[i] = current_a[p[i]-1]
        
        is_lex_smaller = False
        for i in range(n):
            if next_a[i] < best_a[i]:
                is_lex_smaller = True
                break
            elif next_a[i] > best_a[i]:
                break
                
        if is_lex_smaller:
            best_a = list(next_a)
            
        current_a = next_a
        if current_a == initial_a:
            break
            
    print(*(best_a))

if __name__ == '__main__':
    solve()