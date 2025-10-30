import sys

def main():
    N = int(sys.stdin.readline())
    S = sys.stdin.readline().strip()
    pos = []
    for i in range(N):
        if S[i] == '1':
            pos.append(i)
    m = len(pos)
    a = [pos[i] - i for i in range(m)]
    a.sort()
    median = a[m // 2]
    total = 0
    for num in a:
        total += abs(num - median)
    print(total)

if __name__ == "__main__":
    main()