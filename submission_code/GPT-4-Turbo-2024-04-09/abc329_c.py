def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    S = data[1]
    
    count = 0
    i = 0
    
    while i < N:
        start = i
        while i < N and S[i] == S[start]:
            i += 1
        length = i - start
        count += (length * (length + 1)) // 2
    
    print(count)

if __name__ == "__main__":
    main()