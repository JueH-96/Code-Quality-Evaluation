def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = int(data[1])
    T = list(map(int, data[2:]))

    end_time = 0
    for i in range(N):
        if T[i] >= end_time:
            finish = T[i] + A
        else:
            finish = end_time + A
        end_time = finish
        print(finish)

if __name__ == "__main__":
    main()