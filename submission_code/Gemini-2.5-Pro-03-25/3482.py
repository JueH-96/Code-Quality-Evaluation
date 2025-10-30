import collections
from typing import List

# Define TrieNode class to represent nodes in the Aho-Corasick Trie
class TrieNode:
    def __init__(self):
        # children maps character to the next TrieNode in the trie
        self.children = {} 
        # output stores a list of (length, cost) pairs for words ending exactly at this node.
        # During failure link computation, this list is aggregated to include matches found via failure links.
        self.output = [] 
        # fail points to the TrieNode representing the longest proper suffix of the string ending at this node
        # that is also a prefix of some word in the dictionary.
        self.fail = None 

class Solution:
    """
    Finds the minimum cost to construct the target string by concatenating words from a given list,
    where each word concatenation has an associated cost. This solution uses Dynamic Programming 
    combined with the Aho-Corasick algorithm for efficient pattern matching.
    """
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        """
        Calculates the minimum cost to form the target string.

        Args:
            target: The target string to construct.
            words: A list of strings that can be used for concatenation.
            costs: A list of costs corresponding to each word in `words`.

        Returns:
            The minimum cost to form the target string, or -1 if it's impossible.
        """

        # Step 1: Pre-process words and costs to find the minimum cost for each unique word.
        # This efficiently handles cases where the same word appears multiple times with different costs.
        min_costs = {}
        for i in range(len(words)):
            word = words[i]
            cost = costs[i]
            # Store the minimum cost found so far for this word string.
            if word not in min_costs or cost < min_costs[word]:
                min_costs[word] = cost

        # Step 2: Build the Aho-Corasick Trie (Keyword Tree).
        root = TrieNode()
        # Insert each unique word with its minimum cost into the Trie.
        for word, cost in min_costs.items():
            node = root
            # Traverse the Trie, creating nodes as necessary for each character.
            for char in word:
                node = node.children.setdefault(char, TrieNode())
            # At the terminal node corresponding to the end of the word, 
            # store the word's length and its associated minimum cost.
            node.output.append((len(word), cost))

        # Step 3: Compute Failure Links using Breadth-First Search (BFS).
        # Failure links enable efficient transitions in the automaton when a character match fails.
        # During this process, we also aggregate the 'output' lists: words ending at the failure state 
        # are suffixes of words ending at the current state, so their matches should be included.
        queue = collections.deque()
        # Initialize failure links for nodes directly connected to the root (depth 1).
        for node in root.children.values():
            queue.append(node)
            node.fail = root # Nodes at depth 1 always fail back to the root.

        # Process nodes level by level using BFS to compute failure links for all nodes.
        while queue:
            current_node = queue.popleft()
            
            # For each child ('next_node') connected via character 'key':
            for key, next_node in current_node.children.items():
                queue.append(next_node) # Add the child node to the queue for later processing.
                fail_node = current_node.fail # Start finding failure link from the parent's failure node.
                
                # Traverse up the failure links from the parent's fail node until we find a node 
                # that has a child transition for 'key', or until we reach the root.
                while fail_node is not None and key not in fail_node.children:
                    fail_node = fail_node.fail
                
                # Set the failure link for 'next_node'. 
                # If a suitable ancestor node ('fail_node') was found, link to its child via 'key'.
                # Otherwise (if fail_node is None), link back to the root.
                next_node.fail = fail_node.children[key] if fail_node else root
                
                # Aggregate output lists. This is a key step in Aho-Corasick.
                # It ensures that when we reach 'next_node', its 'output' list contains information 
                # not just for words ending exactly at this node, but also for all words that are suffixes 
                # and are identified via the failure path.
                next_node.output.extend(next_node.fail.output) 

        # Step 4: Dynamic Programming using the Aho-Corasick automaton.
        N = len(target)
        # dp[i] will store the minimum cost to construct the prefix target[:i].
        # Initialize dp array: dp[0] = 0 (cost of empty prefix is 0). All other costs are infinity initially.
        dp = [0] + [float('inf')] * N 

        current_node = root # Start the automaton state machine at the root.
        # Iterate through the target string character by character, from index 0 to N-1.
        for i in range(N): 
            char = target[i]
            
            # Perform state transition in the Aho-Corasick automaton based on the current character.
            # If the character `char` does not have a direct transition from `current_node`,
            # follow the failure links until a node with a transition for `char` is found, or root is reached.
            while current_node is not None and char not in current_node.children:
                 current_node = current_node.fail

            if current_node is None:
                 # If we fall back past the root (should ideally not happen with root.fail=None setup),
                 # or if `char` is not a child transition from root itself.
                 # Reset the state to root. Essentially, no ongoing match continues with `char`.
                 current_node = root
            else:
                 # A valid state `current_node` (possibly after following fail links) was found 
                 # that has `char` as a child. Make the transition to the child node.
                 current_node = current_node.children[char]
            
            # After the state transition, `current_node` represents the state after processing target[i].
            # Check all matches ending at this index `i`. The `current_node.output` list contains 
            # all necessary (Length, Cost) pairs due to the aggregation step during failure link computation.
            for length, cost in current_node.output:
                # A word of `length` ending at index `i` corresponds to the substring target[i-length+1 : i+1].
                # To form target[:i+1] using this word, we must have already formed the prefix target[:i-length+1].
                # The minimum cost for this prefix is stored in dp[i - length + 1].
                
                # Check if the required prefix `target[:i-length+1]` is actually reachable (has a finite cost).
                if dp[i - length + 1] != float('inf'):
                    # If reachable, we have found a potential way to form target[:i+1].
                    # Update its minimum cost (stored in dp[i+1]) if this path is cheaper.
                    dp[i + 1] = min(dp[i + 1], dp[i - length + 1] + cost)

        # Step 5: Return the final result.
        # After processing the entire target string, dp[N] holds the minimum cost to form target[:N].
        final_cost = dp[N]
        
        # If final_cost remains infinity, it means the target string cannot be constructed using the given words.
        return final_cost if final_cost != float('inf') else -1