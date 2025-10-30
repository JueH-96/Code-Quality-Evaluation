import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    results = []
    for _ in range(T):
        N, M = int(data[idx]), int(data[idx+1])
        idx += 2
        boxes = []
        for _ in range(N):
            V = int(data[idx])
            P = int(data[idx+1])
            idx += 2
            boxes.append((V, P))
        profitable = []
        for V, P in boxes:
            if V > P:
                profitable.append(V - P)
        profitable.sort(reverse=True)
        k = min(M, len(profitable))
        total = sum(profitable[:k])
        results.append(total)
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == '__main__':
    main()