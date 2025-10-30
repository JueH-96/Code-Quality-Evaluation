def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    N = int(data[0])
    # Read A_1, A_2, ..., A_(N-1)
    A = list(map(int, data[1:]))
    # The overall points sum remains 0 (each game adds +1 and -1)
    # Therefore, score of person N is -sum of scores of persons 1 to N-1.
    result = -sum(A)
    
    print(result)

if __name__ == '__main__':
    main()