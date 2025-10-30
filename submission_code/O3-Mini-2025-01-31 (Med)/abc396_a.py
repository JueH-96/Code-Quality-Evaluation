def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    answer = "No"
    for i in range(N - 2):
        if A[i] == A[i + 1] == A[i + 2]:
            answer = "Yes"
            break
    print(answer)

if __name__ == '__main__':
    main()