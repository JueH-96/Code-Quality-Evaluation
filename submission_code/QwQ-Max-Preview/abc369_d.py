def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    even = 0
    odd = -float('inf')
    
    for a in A:
        new_even = max(even, (odd + 2 * a) if odd != -float('inf') else -float('inf'))
        new_odd = max(even + a, odd)
        even, odd = new_even, new_odd
    
    print(max(even, odd))

if __name__ == "__main__":
    main()