def main():
    import sys
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    last_occurrence = {}
    B = []

    for i in range(N):
        val = A[i]
        if val in last_occurrence:
            B.append(last_occurrence[val])
        else:
            B.append(-1)
        last_occurrence[val] = i + 1

    print(" ".join(map(str, B)))

# Do not remove the call to main() or you will not be awarded any points
main()