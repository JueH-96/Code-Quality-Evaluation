# YOUR CODE HERE
def solve():
    n = int(input())
    
    def to_base_5(n):
        if n == 0:
            return "0"
        nums = []
        while n:
            nums.append(str(n % 5))
            n //= 5
        return "".join(nums[::-1])

    base5 = to_base_5(n - 1)
    ans = ""
    for digit in base5:
        ans += str(int(digit) * 2)
    print(ans)

solve()