class Solution:
    def stringIndices(self, wordsContainer: List[str], wordsQuery: List[str]) -> List[int]:
        # Compute the global best candidate (shortest length, earliest index)
        global_best_length = float('inf')
        global_best_index = 0
        for i, word in enumerate(wordsContainer):
            length = len(word)
            if length < global_best_length or (length == global_best_length and i < global_best_index):
                global_best_length = length
                global_best_index = i
        
        # Build the trie
        root = {'children': {}, 'best': None}
        for index, word in enumerate(wordsContainer):
            reversed_word = word[::-1]
            current_node = root
            for char in reversed_word:
                if char not in current_node['children']:
                    current_node['children'][char] = {'children': {}, 'best': None}
                current_node = current_node['children'][char]
                current_length = len(word)
                current_best = current_node['best']
                if current_best is None:
                    current_node['best'] = (index, current_length)
                else:
                    best_idx, best_len = current_best
                    if current_length < best_len or (current_length == best_len and index < best_idx):
                        current_node['best'] = (index, current_length)
        
        # Process each query
        result = []
        for query in wordsQuery:
            reversed_q = query[::-1]
            current_node = root
            current_best = (global_best_index, global_best_length)
            for char in reversed_q:
                if char in current_node['children']:
                    current_node = current_node['children'][char]
                    current_best = current_node['best']
                else:
                    break
            result.append(current_best[0])
        
        return result