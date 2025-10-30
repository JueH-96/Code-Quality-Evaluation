class TrieNode:
    def __init__(self):
        self.children = [None, None]

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, num):
        node = self.root
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            if not node.children[bit]:
                node.children[bit] = TrieNode()
            node = node.children[bit]
    
    def max_xor(self, num):
        if not self.root.children[0] and not self.root.children[1]:
            return 0
        node = self.root
        max_xor = 0
        for i in range(31, -1, -1):
            bit = (num >> i) & 1
            toggled_bit = 1 - bit
            if node.children[toggled_bit]:
                max_xor |= (1 << i)
                node = node.children[toggled_bit]
            else:
                node = node.children[bit] if node.children[bit] else node
        return max_xor

def precompute_max_xor(arr):
    m = len(arr)
    max_xor = [[0] * m for _ in range(m)]
    for a in range(m):
        trie = Trie()
        current_max = 0
        for b in range(a, m):
            num = arr[b]
            if b > a:
                current_xor = trie.max_xor(num)
                current_max = max(current_max, current_xor)
            trie.insert(num)
            max_xor[a][b] = current_max
    return max_xor

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        prefix_xor = [0] * (n + 1)
        for i in range(n):
            prefix_xor[i + 1] = prefix_xor[i] ^ nums[i]
        
        even_E = []
        odd_E = []
        for i in range(n + 1):
            if i % 2 == 0:
                even_E.append(prefix_xor[i])
            else:
                odd_E.append(prefix_xor[i])
        
        max_xor_even_E = precompute_max_xor(even_E)
        max_xor_odd_E = precompute_max_xor(odd_E)
        
        even_nums = []
        odd_nums = []
        for i in range(n):
            if i % 2 == 0:
                even_nums.append(nums[i])
            else:
                odd_nums.append(nums[i])
        
        max_xor_even_O = precompute_max_xor(even_nums)
        max_xor_odd_O = precompute_max_xor(odd_nums)
        
        res = []
        for L, R in queries:
            even_max = 0
            start_j_even = (L + 1) // 2
            end_j_even = (R + 1) // 2
            a_even = max(0, start_j_even)
            b_even = min(len(even_E) - 1, end_j_even)
            even_max_even = max_xor_even_E[a_even][b_even] if a_even <= b_even else 0
            
            start_j_odd = (L) // 2
            end_j_odd = (R) // 2
            a_odd = max(0, start_j_odd)
            b_odd = min(len(odd_E) - 1, end_j_odd)
            even_max_odd = max_xor_odd_E[a_odd][b_odd] if a_odd <= b_odd else 0
            even_max = max(even_max_even, even_max_odd)
            
            start_even_num = (L + 1) // 2
            end_even_num = R // 2
            a_even_num = max(0, start_even_num)
            b_even_num = min(len(even_nums) - 1, end_even_num)
            odd_max_even = max_xor_even_O[a_even_num][b_even_num] if a_even_num <= b_even_num else 0
            
            start_odd_num = (L) // 2
            end_odd_num = (R - 1) // 2
            a_odd_num = max(0, start_odd_num)
            b_odd_num = min(len(odd_nums) - 1, end_odd_num)
            odd_max_odd = max_xor_odd_O[a_odd_num][b_odd_num] if a_odd_num <= b_odd_num else 0
            odd_max = max(odd_max_even, odd_max_odd)
            
            res.append(max(even_max, odd_max))
        
        return res