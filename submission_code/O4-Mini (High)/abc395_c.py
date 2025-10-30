import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    last = {}
    ans = N + 1
    for i, x in enumerate(A):
        if x in last:
            # candidate subarray from last[x] to i (inclusive)
            ans = min(ans, i - last[x] + 1)
        last[x] = i
    # if ans was never updated, it's > N
    print(ans if ans <= N else -1)

if __name__ == "__main__":
    main()