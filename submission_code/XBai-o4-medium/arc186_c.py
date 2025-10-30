import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    T = int(input[idx])
    idx += 1
    results = []
    
    for _ in range(T):
        N, M = int(input[idx]), int(input[idx+1])
        idx += 2
        values = []
        for _ in range(N):
            V = int(input[idx])
            P = int(input[idx+1])
            idx += 2
            profit = V - P - 1
            if profit > 0:
                values.append(profit)
        # Take all positive values, sum them
        total = sum(values)
        results.append(max(0, total))
    
    for res in results:
        print(res)

if __name__ == '__main__':
    main()