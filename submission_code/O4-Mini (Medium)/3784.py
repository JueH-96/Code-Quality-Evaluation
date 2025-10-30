from typing import List

class Solution:
    def longestCommonPrefix(self, words: List[str], k: int) -> List[int]:
        n = len(words)
        # Edge case: if after removal fewer than k words remain, always 0
        if k > n - 1:
            return [0] * n

        # Build trie with node list
        # Each node: children dict, count, depth
        nodes = [{'children': {}, 'cnt': 0, 'depth': 0}]
        # For each word, record the path of node indices for its prefixes
        word_paths = []  # list of lists of node indices

        # Build trie and counts
        for w in words:
            cur = 0
            path = []
            for ch in w:
                child = nodes[cur]['children'].get(ch)
                if child is None:
                    # create a new node
                    child = len(nodes)
                    nodes[cur]['children'][ch] = child
                    nodes.append({'children': {}, 'cnt': 0, 'depth': nodes[cur]['depth'] + 1})
                cur = child
                nodes[cur]['cnt'] += 1
                path.append(cur)
            word_paths.append(path)

        # Determine maximum depth in trie
        max_depth = 0
        for node in nodes:
            if node['depth'] > max_depth:
                max_depth = node['depth']

        # Compute C[L]: number of prefixes at depth L with count >= k
        # Also track which depths might become invalid (C[L]==1 and exactly that prefix had cnt==k)
        C = [0] * (max_depth + 1)  # 0..max_depth
        K = [0] * (max_depth + 1)  # number of prefixes at depth L with cnt == k
        # Walk all nodes
        for node in nodes:
            d = node['depth']
            cnt = node['cnt']
            if d > 0 and cnt >= k:
                C[d] += 1
                if cnt == k:
                    K[d] += 1

        # Candidate depths where at least one prefix has count >= k
        candidate_depths = [d for d in range(1, max_depth + 1) if C[d] > 0]
        # If no candidate depths, all answers are 0
        if not candidate_depths:
            return [0] * n

        # For each word, collect its "invalid" depths:
        # those depths d where along its path the node has cnt==k and C[d]==1
        # (meaning that prefix is unique among those with cnt>=k, removal kills it)
        res = [0] * n
        # We'll do lookups of invalid depths by turning them into a set per word
        for i, path in enumerate(word_paths):
            invalid = set()
            for node_idx in path:
                node = nodes[node_idx]
                d = node['depth']
                # if this prefix had exactly k occurrences and it was the only prefix of length d
                if node['cnt'] == k and C[d] == 1:
                    invalid.add(d)
            # From the largest candidate depth down, pick the first not invalid
            ans = 0
            for d in reversed(candidate_depths):
                if d not in invalid:
                    ans = d
                    break
            res[i] = ans

        return res