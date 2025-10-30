MOD = 10**9 + 7

def min_cost(nums, cost1, cost2):
    max_num = max(nums)
    d = [max_num - x for x in nums]
    S = sum(d)
    m = sum(1 for x in d if x > 0)
    if m == 0:
        current_cost = 0
    else:
        if m >= 2:
            K = S // 2
        else:
            K = 0
        current_cost = K * cost2 + (S - 2 * K) * cost1
    return current_cost % MOD

# Read input
import sys

def main():
    input = sys.stdin.read().split()
    idx = 0
    nums = list(map(int, input[idx:idx+5]))
    idx +=5
    cost1 = int(input[idx])
    cost2 = int(input[idx+1])
    print(min_cost(nums, cost1, cost2))

if __name__ == "__main__":
    main()