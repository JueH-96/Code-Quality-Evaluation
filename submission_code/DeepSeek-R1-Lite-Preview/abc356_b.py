import sys

def main():
    data = sys.stdin.read().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    goals = []
    for _ in range(M):
        goals.append(int(data[idx]))
        idx += 1
    totals = [0] * M
    for _ in range(N):
        for j in range(M):
            totals[j] += int(data[idx])
            idx += 1
    for i in range(M):
        if totals[i] < goals[i]:
            print("No")
            return
    print("Yes")

if __name__ == "__main__":
    main()