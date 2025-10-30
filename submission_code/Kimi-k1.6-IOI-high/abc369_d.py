def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    n = int(data[0])
    A = list(map(int, data[1:n+1]))
    
    even = 0
    odd = float('-inf')
    
    for a in A:
        new_even = max(even, odd + 2 * a)
        new_odd = max(odd, even + a)
        even, odd = new_even, new_odd
    
    print(max(even, odd))

if __name__ == "__main__":
    main()