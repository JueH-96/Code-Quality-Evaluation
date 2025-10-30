# YOUR CODE HERE
import sys
from collections import defaultdict

def min_cost_to_form_string(T, N, bags):
    dp = defaultdict(lambda: float('inf'))
    dp[0] = 0
    
    for i in range(N):
        new_dp = defaultdict(lambda: float('inf'))
        for j in range(len(T) + 1):
            if dp[j] != float('inf'):
                new_dp[j] = min(new_dp[j], dp[j])
                for s in bags[i]:
                    if T[j:j + len(s)] == s:
                        new_dp[j + len(s)] = min(new_dp[j + len(s)], dp[j] + 1)
        dp = new_dp
    
    return dp[len(T)] if dp[len(T)] != float('inf') else -1

def main():
    input = sys.stdin.read
    data = input().split()
    
    T = data[0]
    N = int(data[1])
    index = 2
    bags = []
    
    for _ in range(N):
        A = int(data[index])
        index += 1
        bag = data[index:index + A]
        index += A
        bags.append(bag)
    
    print(min_cost_to_form_string(T, N, bags))

if __name__ == "__main__":
    main()