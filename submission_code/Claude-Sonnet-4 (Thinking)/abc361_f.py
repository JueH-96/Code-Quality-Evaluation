def max_base(N, b):
    # Find the maximum a such that a^b <= N using binary search
    left, right = 1, int(N**(1.0/b)) + 10
    while left < right:
        mid = (left + right + 1) // 2
        if mid**b <= N:
            left = mid
        else:
            right = mid - 1
    return left

N = int(input())

powers = set()

for b in range(2, 61):  # log2(10^18) < 60
    max_a = max_base(N, b)
    for a in range(1, max_a + 1):
        powers.add(a**b)

print(len(powers))