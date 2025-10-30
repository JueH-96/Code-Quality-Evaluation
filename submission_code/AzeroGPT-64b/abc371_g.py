from typing import List
from collections import deque

def input_to_intlist():
    return list(map(int, input().split()))

def input_and_process():
    n = int(input())
    p = input_to_intlist()
    a = input_to_intlist()
    return n, p, a

class TreeNode:
    def __init__(self, data):
        self.data = data
        self.children = []
        self.descendants = set()
        self.parent = None

class Solution:
    def get_fixed_points(self, n: int, p: List[int]) -> List[int]:
        return [i for i in range(n) if i == p[i]]

    def build_tree(self, p: List[int]) -> TreeNode:
        root = TreeNode(0)
        node_map = {0: root}
        
        def dfs(node_index):
            node = node_map[node_index]
            child_index = p[node_index]
            if child_index not in node_map:
                child_node = TreeNode(child_index)
                node_map[child_index] = child_node
                child_node.parent = node
                dfs(child_index)
            else:
                child_node = node_map[child_index]
            node.children.append(child_node)
            
        for i in range(1, len(p)):
            dfs(i)
        
        return root

    def traverse_tree(self, node: TreeNode):
        result = deque()
        stack = [node]
        while stack:
            current = stack.pop()
            result.append(current.data)
            for child in current.children:
                stack.append(child)
        return list(result)

    def bfs_traverse_tree(self, node: TreeNode):
        queue = deque([node])
        result = []
        while queue:
            current = queue.popleft()
            result.append(current)
            result.extend(current.children)
            for child in current.children:
                queue.append(child)
        return result

    def calculate_descendants(self, root: TreeNode):
        nodes = self.bfs_traverse_tree(root)
        for node in nodes:
            for child in node.children:
                node.descendants.update(child.descendants)
                node.descendants.add(child.data)
    
    def update_descendants(self, root: TreeNode, a: List[int]):
        nodes = self.bfs_traverse_tree(root)
        for node in nodes:
            if node.parent is not None:
                index = a.index(node.parent.data)
                node.descendants.add(a[index])
        
    def update_tree(self, root: TreeNode, a: List[int]):
        nodes = self.bfs_traverse_tree(root)
        mini = {}
        for node in nodes:
            mini[node.data] = min([a[i] for i in node.descendants])
        for node in nodes:
            if node.parent is not None:
                value = mini[node.parent.data]
                if value in node.descendants:
                    mini[node.data] = value
        return mini

    def generate_min_list(self, mini: dict, nodes: List[TreeNode], a: List[int]):
        for node in nodes:
            value = mini[node.data]
            index = a.index(node.data)
            a[index] = value
    
    def smallest_permutation(self, n: int, p: List[int], a: List[int]) -> List[int]:
        fixed_points = self.get_fixed_points(n, p)
        non_fixed_points = [i for i in range(n) if i not in fixed_points]
        root = self.build_tree(p)
        self.calculate_descendants(root)
        self.update_descendants(root, a)
        mini = self.update_tree(root, a)
        nodes = self.bfs_traverse_tree(root)
        self.generate_min_list(mini, nodes, a)
        for fixed in fixed_points:
            a[fixed] = min(a[fixed], mini[fixed])
        return a

# Solve the problem
solution = Solution()
n, p, a = input_and_process()
result = solution.smallest_permutation(n, p, a)
print(*result)