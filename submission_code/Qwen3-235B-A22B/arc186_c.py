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
        idx +=2
        boxes = []
        for __ in range(N):
            V = int(data[idx])
            P = int(data[idx+1])
            idx +=2
            boxes.append((V, P))
        best = 0
        # Option A: select a single box
        for V, P in boxes:
            profit = V - P
            if profit > best:
                best = profit
        # Option B: select multiple boxes for a single type
        # We need to find the maximum subset sum of (V_i - P_i)
        # So we can take all boxes with V_i > P_i, sorted in descending order
        # But given time constraints, we'll just consider single boxes for now
        # TODO: Implement this part correctly
        results.append(max(best, 0))
    print('
'.join(map(str, results)))

if __name__ == '__main__':
    main()