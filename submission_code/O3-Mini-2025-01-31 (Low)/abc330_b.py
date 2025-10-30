def main():
    import sys
    input_data = sys.stdin.read().split()
    N = int(input_data[0])
    L = int(input_data[1])
    R = int(input_data[2])
    A = list(map(int, input_data[3:]))
    
    results = []
    for a in A:
        if a < L:
            results.append(L)
        elif a > R:
            results.append(R)
        else:
            results.append(a)
    
    print(" ".join(map(str, results)))

if __name__ == '__main__':
    main()