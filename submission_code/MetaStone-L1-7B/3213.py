import bisect

class FenwickTree:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 1)
    
    def update(self, index, delta=1):
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= index & -index
        return res

def count_subarrays(nums, k):
    if not nums:
        return 0
    max_val = max(nums)
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + (1 if nums[i] == max_val else 0)
    
    ft = FenwickTree(n)
    ft.update(prefix[0])
    result = 0
    
    for j in range(n):
        current_prefix = prefix[j + 1]
        target = current_prefix - k
        count = ft.query(target)
        result += count
        ft.update(current_prefix)
    
    return result

# Read input
import sys
def main():
    input = sys.stdin.read().split()
    ptr = 0
    nums = list(map(int, input[ptr:ptr + len(input)//2]))
    ptr += len(input)//2
    k = int(input[ptr])
    ptr += 1
    print(count_subarrays(nums, k))

if __name__ == "__main__":
    main()