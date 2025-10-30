def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    
    even = 0
    odd = float('-inf')
    
    for a in A:
        even, odd = max(even, odd + 2 * a), max(odd, even + a)
    
    print(max(even, odd))

if __name__ == "__main__":
    main()