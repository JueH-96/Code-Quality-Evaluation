from collections import deque
from typing import List

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        if n == 0:
            return []
        max_depth_val = max(len(word) for word in words)
        
        class TrieNode:
            __slots__ = ['count', 'children']
            def __init__(self):
                self.count = 0
                self.children = {}
        
        root = TrieNode()
        for word in words:
            node = root
            node.count += 1
            for c in word:
                if c not in node.children:
                    node.children[c] = TrieNode()
                node = node.children[c]
                node.count += 1
        
        size = max_depth_val + 1
        depth_counter = [0] * size
        q = deque()
        q.append((root, 0))
        while q:
            node, depth = q.popleft()
            if depth < size:
                if node.count >= k:
                    depth_counter[depth] += 1
            for c, child in node.children.items():
                q.append((child, depth + 1))
        
        seg_tree = [-1] * (4 * size)
        
        def update_seg_tree(idx, l, r, pos, val):
            if l == r:
                seg_tree[idx] = val
                return
            mid = (l + r) // 2
            if pos <= mid:
                update_seg_tree(2 * idx + 1, l, mid, pos, val)
            else:
                update_seg_tree(2 * idx + 2, mid + 1, r, pos, val)
            right_val = seg_tree[2 * idx + 2]
            left_val = seg_tree[2 * idx + 1]
            if right_val != -1:
                seg_tree[idx] = right_val
            else:
                seg_tree[idx] = left_val
        
        for d in range(size):
            if depth_counter[d] > 0:
                update_seg_tree(0, 0, size - 1, d, d)
            else:
                update_seg_tree(0, 0, size - 1, d, -1)
        
        ans = [0] * n
        
        for i in range(n):
            word = words[i]
            stack = []
            node = root
            old_count = node.count
            node.count -= 1
            stack.append((node, 0, old_count))
            for j, c in enumerate(word):
                node = node.children[c]
                old_count = node.count
                node.count -= 1
                depth = j + 1
                stack.append((node, depth, old_count))
            
            for node, depth, old_count in stack:
                new_count = old_count - 1
                was_valid = (old_count >= k)
                now_valid = (new_count >= k)
                if was_valid and not now_valid:
                    depth_counter[depth] -= 1
                    if depth_counter[depth] == 0:
                        update_seg_tree(0, 0, size - 1, depth, -1)
            
            candidate = seg_tree[0]
            if candidate == -1:
                ans[i] = 0
            else:
                ans[i] = candidate
            
            for node, depth, old_count in reversed(stack):
                node.count = old_count
            for node, depth, old_count in stack:
                new_count = old_count - 1
                was_valid = (old_count >= k)
                now_valid = (new_count >= k)
                if was_valid and not now_valid:
                    depth_counter[depth] += 1
                    if depth_counter[depth] == 1:
                        update_seg_tree(0, 0, size - 1, depth, depth)
        
        return ans