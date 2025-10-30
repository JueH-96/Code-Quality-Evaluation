N = int(input())
A = list(map(int,input().split()))
def solve():
    from collections import defaultdict,Counter
    itemList = list(i + 1 for i in range(N))
    for index in range(N):
        itemList[index] = (A[index],index + 1)
    itemList.sort(key = lambda x: x[0])
    ans = 0
    value_position_dict = defaultdict(lambda: Counter())
    for index in range(N):
        value, pos = itemList[index]
        tmp_ans = 0
        in_between = value_position_dict[value]
        for p in range(1,max(in_between.keys()) + 1 if in_between.keys() else 0):
            k_value = in_between[p]
            tmp_ans += k_value * (pos - p - k_value)
        ans += tmp_ans * (N - pos - in_between[pos])
        value_position_dict[value][pos] += 1
    return ans
print(solve())