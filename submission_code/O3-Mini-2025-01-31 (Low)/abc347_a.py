def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    K = int(input_data[1])
    A = list(map(int, input_data[2:2+N]))
    
    result = []
    for x in A:
        if x % K == 0:
            result.append(x // K)
    
    result.sort()
    print(" ".join(map(str, result)))

if __name__ == '__main__':
    main()