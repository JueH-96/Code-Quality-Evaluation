from collections import deque

class Line:
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def value(self, x):
        return self.a * x + self.b

    def intersect_x(self, other):
        if self.a == other.a:
            return float('inf')
        return (other.b - self.b) / (self.a - other.a)

    def is_better_than(self, other, x):
        return self.value(x) > other.value(x)

def add_line(deque, new_line):
    while len(deque) >= 2:
        l1 = deque[-2]
        l2 = deque[-1]
        x1 = l1.intersect_x(l2)
        x2 = l1.intersect_x(new_line)
        if x1 <= x2:
            deque.pop()
        else:
            break
    deque.append(new_line)

def get_max(deque, x):
    while len(deque) >= 2:
        l1 = deque[0]
        l2 = deque[1]
        if l1.value(x) > l2.value(x):
            deque.popleft()
        else:
            deque.pop()
    return deque[0].value(x) if deque else -float('inf')

def findMaximumScore(nums: List[int]) -> int:
    n = len(nums)
    if n == 0:
        return 0
    dp = [0] * n
    dq = deque()
    a0 = nums[0]
    b0 = dp[0] - 0 * a0
    dq.append(Line(a0, b0))
    for i in range(1, n):
        current_max = get_max(dq, i)
        dp[i] = current_max
        a_i = nums[i]
        b_i = dp[i] - i * a_i
        add_line(dq, Line(a_i, b_i))
    return dp[-1]