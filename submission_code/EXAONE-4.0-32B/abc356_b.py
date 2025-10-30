def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+m]))
    totals = [0] * m
    index = 2 + m
    for _ in range(n):
        row = list(map(int, data[index:index+m]))
        index += m
        for j in range(m):
            totals[j] += row[j]
    met = True
    for j in range(m):
        if totals[j] < A[j]:
            met = False
            break
    print("Yes" if met else "No")

if __name__ == "__main__":
    main()