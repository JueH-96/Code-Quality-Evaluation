def main():
    import sys
    input = sys.stdin.read().split()
    n = int(input[0])
    H = list(map(int, input[1:n+1]))
    
    result = []
    prev = 0
    for h in H:
        current = prev + h + 1
        result.append(current)
        prev = current
    
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()