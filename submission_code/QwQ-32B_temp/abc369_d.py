import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    
    prev_even = 0
    prev_odd = -float('inf')
    
    for a in A:
        new_even = max(prev_even, prev_odd + 2 * a)
        new_odd = max(prev_odd, prev_even + a)
        prev_even, prev_odd = new_even, new_odd
    
    print(max(prev_even, prev_odd))

if __name__ == "__main__":
    main()