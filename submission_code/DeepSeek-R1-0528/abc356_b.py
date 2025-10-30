def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    m = int(data[1])
    A = list(map(int, data[2:2+m]))
    index = 2 + m
    total = [0] * m
    for _ in range(n):
        food_data = list(map(int, data[index:index+m]))
        index += m
        for j in range(m):
            total[j] += food_data[j]
    all_met = True
    for j in range(m):
        if total[j] < A[j]:
            all_met = False
            break
    print("Yes" if all_met else "No")

if __name__ == "__main__":
    main()