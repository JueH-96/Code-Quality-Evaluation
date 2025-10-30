import sys

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    intervals = []
    for i in range(N):
        l = int(data[1 + 2*i])
        r = int(data[2 + 2*i])
        intervals.append((l, r))
    sorted_by_l = sorted(intervals, key=lambda x: x[0])
    sorted_by_r = sorted(intervals, key=lambda x: x[1])
    ans = 0
    active = 0
    j = 0
    for i in range(N):
        l_i = sorted_by_l[i][0]
        while j < N and sorted_by_r[j][1] < l_i:
            active -=1
            j +=1
        ans += active
        active +=1
    print(ans)

if __name__ == "__main__":
    main()