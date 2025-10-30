import collections
from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Build Aho-Corasick automaton
        root = 0
        nodes = [{'children': {}, 'is_end': False, 'pattern_len': -1}]  # Root node
        
        # Insert all forbidden patterns into the trie
        for pattern in forbidden:
            node_idx = 0
            for char in pattern:
                if char not in nodes[node_idx]['children']:
                    new_idx = len(nodes)
                    nodes.append({'children': {}, 'is_end': False, 'pattern_len': -1})
                    nodes[node_idx]['children'][char] = new_idx
                node_idx = nodes[node_idx]['children'][char]
            nodes[node_idx]['is_end'] = True
            nodes[node_idx]['pattern_len'] = len(pattern)
        
        # Set failure and output for root
        nodes[root]['failure'] = root
        nodes[root]['output_min_len'] = float('inf')
        
        # Create transition table
        num_nodes = len(nodes)
        trans = [[0] * 26 for _ in range(num_nodes)]
        # Set transitions for root
        for c_idx in range(26):
            char = chr(c_idx + ord('a'))
            if char in nodes[root]['children']:
                trans[root][c_idx] = nodes[root]['children'][char]
            else:
                trans[root][c_idx] = root  # Stay at root
        
        # BFS to set failure links, output_min_len, and transitions for other nodes
        queue = collections.deque()
        # Enqueue all children of root and set their failure and output_min_len
        for char, child_idx in nodes[root]['children'].items():
            nodes[child_idx]['failure'] = root
            if nodes[child_idx]['is_end']:
                nodes[child_idx]['output_min_len'] = nodes[child_idx]['pattern_len']
            else:
                nodes[child_idx]['output_min_len'] = nodes[root]['output_min_len']  # inf
            queue.append(child_idx)
        
        while queue:
            current_idx = queue.popleft()
            # Set transitions for current node
            for c_idx in range(26):
                char = chr(c_idx + ord('a'))
                if char in nodes[current_idx]['children']:
                    next_state = nodes[current_idx]['children'][char]
                    trans[current_idx][c_idx] = next_state
                else:
                    failure = nodes[current_idx]['failure']
                    trans[current_idx][c_idx] = trans[failure][c_idx]
            
            # Process children of current node
            for char, next_idx in nodes[current_idx]['children'].items():
                # Compute failure for next_idx
                state = nodes[current_idx]['failure']
                while state != root and char not in nodes[state]['children']:
                    state = nodes[state]['failure']
                if char in nodes[state]['children']:
                    fail_state = nodes[state]['children'][char]
                else:
                    fail_state = root
                nodes[next_idx]['failure'] = fail_state
                
                # Set output_min_len for next_idx
                if nodes[next_idx]['is_end']:
                    min_len = nodes[next_idx]['pattern_len']
                else:
                    min_len = float('inf')
                fail_output_min = nodes[fail_state]['output_min_len']
                nodes[next_idx]['output_min_len'] = min(min_len, fail_output_min)
                
                # Enqueue next_idx
                queue.append(next_idx)
        
        # Sliding window to find longest valid substring
        current_state = root
        left = 0
        max_len = 0
        
        for right in range(len(word)):
            char_idx = ord(word[right]) - ord('a')
            # Move to next state using transition table
            current_state = trans[current_state][char_idx]
            
            # Check if there is a forbidden substring ending at right
            if nodes[current_state]['output_min_len'] < float('inf'):
                min_len_match = nodes[current_state]['output_min_len']
                max_start_idx = right - min_len_match + 1
                left = max(left, max_start_idx + 1)
            
            # Calculate the length of the current valid window
            len_window = max(0, right - left + 1)
            if len_window > max_len:
                max_len = max(max_len, len_window)
        
        return max_len