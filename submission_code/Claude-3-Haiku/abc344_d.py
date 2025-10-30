from collections import defaultdict

def min_cost(target, n, bags):
    dp = [float('inf')] * (len(target) + 1)
    dp[0] = 0

    for i in range(1, len(target) + 1):
        for j in range(n):
            for s in bags[j]:
                if target[i - len(s):i] == s:
                    dp[i] = min(dp[i], dp[i - len(s)] + 1)

    return dp[len(target)] if dp[len(target)] != float('inf') else -1

def main():
    target = input()
    n = int(input())
    bags = defaultdict(list)
    for i in range(n):
        line = input().split()
        a = int(line[0])
        for j in range(1, a + 1):
            bags[i].append(line[j])
    print(min_cost(target, n, bags))

if __name__ == "__main__":
    main()