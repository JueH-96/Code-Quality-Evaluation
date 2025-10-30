# YOUR CODE HERE
def min_cost_to_form_string(T, N, bags):
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0

    for i in range(N):
        new_dp = dp.copy()
        for string in bags[i]:
            for j in range(len(T) - len(string) + 1):
                if T[j:j+len(string)] == string:
                    new_dp[j+len(string)] = min(new_dp[j+len(string)], dp[j] + 1)
        dp = new_dp

    return dp[len(T)] if dp[len(T)] != float('inf') else -1

T = input().strip()
N = int(input())
bags = []
for _ in range(N):
    bag = input().split()
    bags.append(bag[1:])

result = min_cost_to_form_string(T, N, bags)
print(result)