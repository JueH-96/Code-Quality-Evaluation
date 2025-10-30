def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 1:
        print(1)
        return
        
    total = 1  # for the first element (index0)
    total += 2  # for index1: two APs: [1] and [0,1]
    prev = 2    # dp for index1: the longest AP ending at index1 has length 2

    for i in range(2, n):
        if A[i] - A[i-1] == A[i-1] - A[i-2]:
            prev = prev + 1
        else:
            prev = 2
        total += prev
        
    print(total)

if __name__ == "__main__":
    main()