import sys
from typing import List

sys.setrecursionlimit(100005)

class Solution:
    def findAnswer(self, parent: List[int], s: str) -> List[bool]:
        n = len(parent)
        # Build children list
        children = [[] for _ in range(n)]
        for i in range(1, n):
            p = parent[i]
            children[p].append(i)
        for child_list in children:
            child_list.sort()  # Sort to have increasing order

        # Define char to int mapping
        def char_to_int(c):
            return ord(c) - ord('a') + 1

        base = 31
        mod = 1000000007

        # Answer array
        ans = [False] * n

        # DFS function
        def dfs(x):
            # Get children of x
            childs = children[x]
            k = len(childs)
            m = k + 1  # Number of blocks

            # Recursive call on children to get their data
            block_data = []
            len_sum_children = 0
            for y in childs:
                len_y, hash_f_y, hash_r_y, is_p_y = dfs(y)
                block_data.append((len_y, hash_f_y, hash_r_y))
                len_sum_children += len_y

            # Add c_x block
            c_x_val = char_to_int(s[x])
            c_x_block = (1, c_x_val, c_x_val)  # Len, hash_f, hash_r for c_x

            # Blocks list: children's blocks + c_x block
            blocks = block_data + [c_x_block]

            # Compute is_palindrome
            is_palindrome = True
            for i in range(m):
                b_i_len, b_i_hash_r = blocks[i][0], blocks[i][2]  # Len and hash_r of B_i
                j = m - 1 - i
                b_j_len, b_j_hash_f = blocks[j][0], blocks[j][1]  # Len and hash_f of B_j
                if b_i_len != b_j_len or b_i_hash_r != b_j_hash_f:
                    is_palindrome = False
                    break

            # Set ans for x
            ans[x] = is_palindrome

            # Compute len_s
            len_s = len_sum_children + 1

            # Compute hash_f_s
            hash_f_s = 0
            for b in blocks:
                len_b = b[0]
                h_f_b = b[1]
                hash_f_s = (hash_f_s * pow(base, len_b, mod) % mod + h_f_b % mod) % mod

            # Compute hash_r_s
            hash_r_s = 0
            for b in reversed(blocks):  # Iterate blocks in reverse order
                len_b = b[0]
                h_r_b = b[2]  # Hash_r of B
                hash_r_s = (hash_r_s * pow(base, len_b, mod) % mod + h_r_b % mod) % mod

            # Return the tuple for x
            return len_s, hash_f_s, hash_r_s, is_palindrome

        # Call DFS from root 0
        dfs(0)
        return ans