def solve(x, y, z, s):
    firstCost, secondCost = min(x, y), max(x, y)
    ans, evenCap, oddCap = 0, z + firstCost, z + secondCost
    for i, w in enumerate(s):
        if i & 1:
            evenCap = min(evenCap, z + firstCost)
            oddCap += w.isupper() * firstCost + w.islower() * secondCost
        else:
            oddCap = min(oddCap, z + secondCost)
            evenCap += w.isupper() * secondCost + w.islower() * firstCost
        ans = min(ans, evenCap, oddCap)
    return ans

x, y, z = map(int, input().split())
s = input()
print(solve(x, y, z, s))