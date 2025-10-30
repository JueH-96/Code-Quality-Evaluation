import sys
input = sys.stdin.read
data = input().split()

T = data[0]
N = int(data[1])
bags = []
index = 2

for i in range(N):
    A_i = int(data[index])
    strings = data[index + 1:index + 1 + A_i]
    bags.append(strings)
    index += 1 + A_i

def can_form_T(T, bags, N):
    dp = [False] * (len(T) + 1)
    dp[0] = True
    for i in range(N):
        new_dp = dp[:]
        for string in bags[i]:
            for j in range(len(T) - len(string), -1, -1):
                if dp[j] and T.startswith(string, j):
                    new_dp[j + len(string)] = True
        dp = new_dp
    return dp[len(T)]

def min_cost_to_form_T(T, bags, N):
    if not can_form_T(T, bags, N):
        return -1

    cost = 0
    current_string = ""

    for i in range(N):
        found = False
        for string in bags[i]:
            if T.startswith(string, len(current_string)):
                current_string += string
                cost += 1
                found = True
                break
        if not found:
            continue

    return cost if current_string == T else -1

result = min_cost_to_form_T(T, bags, N)
print(result)