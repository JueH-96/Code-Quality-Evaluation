def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    D = int(data[1])
    S = data[2]
    
    initial_empty = S.count('.')
    total_empty = initial_empty + D
    print(total_empty)

if __name__ == "__main__":
    main()