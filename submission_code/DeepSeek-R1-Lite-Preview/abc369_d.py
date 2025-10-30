def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    even = 0
    odd = float('-inf')
    
    for a in A:
        new_odd = max(odd, even + a)
        new_even = max(even, odd + 2 * a)
        even, odd = new_even, new_odd
    
    print(int(max(even, odd)))

if __name__ == "__main__":
    main()