import sys

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    A = list(map(int, data[2:2+N]))
    S = set()
    for a in A:
        if 1 <= a <= K:
            S.add(a)
    sum_total = K * (K + 1) // 2
    sum_A = sum(S)
    desired_sum = sum_total - sum_A
    print(desired_sum)

if __name__ == '__main__':
    main()