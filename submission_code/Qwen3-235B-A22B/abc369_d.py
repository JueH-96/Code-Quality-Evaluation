import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    prev_even = 0
    prev_odd = -float('inf')
    
    for a in A:
        curr_even = max(prev_even, prev_odd + 2 * a)
        curr_odd = max(prev_odd, prev_even + a)
        prev_even, prev_odd = curr_even, curr_odd
    
    print(max(prev_even, prev_odd))

if __name__ == "__main__":
    main()