def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    prev_even = 0
    prev_odd = float('-inf')
    
    for a in A:
        new_even = max(prev_even, prev_odd + a * 2)
        new_odd = max(prev_odd, prev_even + a)
        prev_even, prev_odd = new_even, new_odd
    
    print(max(prev_even, prev_odd))

if __name__ == '__main__':
    main()