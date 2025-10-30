import sys

def main():
    input = sys.stdin.read
    data = input().split()
    idx = 0
    T = int(data[idx])
    idx += 1
    for _ in range(T):
        N, M = int(data[idx]), int(data[idx+1])
        idx += 2
        boxes = []
        for _ in range(N):
            V, P = int(data[idx]), int(data[idx+1])
            boxes.append((V, P))
            idx += 2
        # Sort boxes by V in ascending order
        boxes.sort()
        # Precompute the maximum profit for each box
        max_profit = 0
        for V, P in boxes:
            if V >= 1:
                # The profit is V - P
                profit = V - P
                if profit > max_profit:
                    max_profit = profit
        # The total profit is the maximum profit multiplied by M
        total_profit = max_profit * M
        print(total_profit)

if __name__ == "__main__":
    main()