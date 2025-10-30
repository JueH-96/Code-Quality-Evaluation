import sys

def solve():
    a_b_str = sys.stdin.readline().split()
    a = int(a_b_str[0])
    b = int(a_b_str[1])
    
    result = (a + b - 1) // b
    print(result)

if __name__ == '__main__':
    solve()