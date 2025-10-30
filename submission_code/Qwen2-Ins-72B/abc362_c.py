import sys

input = sys.stdin.readline

def main():
    N = int(input())
    LR = [list(map(int, input().split())) for _ in range(N)]

    ans = []
    sum_ = 0
    for i in range(N):
        l, r = LR[i]
        if l <= -r:
            ans.append(-r)
            sum_ += -r
        elif l >= -r:
            ans.append(l)
            sum_ += l
        else:
            ans.append(l)
            sum_ += l
            LR[i][0] = -r

    if sum_ == 0:
        print("Yes")
        print(*ans)
        return

    for i in range(N):
        l, r = LR[i]
        if l <= -sum_ <= r:
            ans[i] = -sum_
            print("Yes")
            print(*ans)
            return

    print("No")

if __name__ == "__main__":
    main()