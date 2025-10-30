def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    count = 0
    
    while True:
        positive = sum(1 for x in A if x > 0)
        if positive <= 1:
            break
        A.sort(reverse=True)
        A[0] -= 1
        A[1] -= 1
        count += 1
    print(count)

if __name__ == "__main__":
    main()