def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    result = []
    for i in range(n-1):
        a, b = A[i], A[i+1]
        step = 1 if b > a else -1
        # range(a, b, step) produces a, a+step, ..., up to but not including b
        for x in range(a, b, step):
            result.append(x)
    # append the last element
    result.append(A[-1])
    
    print(" ".join(map(str, result)))

if __name__ == "__main__":
    main()