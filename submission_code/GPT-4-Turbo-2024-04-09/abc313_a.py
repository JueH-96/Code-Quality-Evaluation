def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    P = list(map(int, data[1:]))
    
    P1 = P[0]
    max_other = max(P[1:])
    
    if P1 > max_other:
        print(0)
    else:
        print(max_other - P1 + 1)

if __name__ == "__main__":
    main()