def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    last_occurrence = {}
    total = 0
    for i in range(N):
        val = A[i]
        current_last = last_occurrence.get(val, 0)
        contribution = (i + 1 - current_last) * (N - i)
        total += contribution
        last_occurrence[val] = i + 1
    print(total)

if __name__ == "__main__":
    main()